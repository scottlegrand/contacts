import cupy as cp

ligand_map = {'A': 0, 'C': 0, 'N': 1, 'NA': 1, 'O': 2, 'OA': 2, 'F': 3, 'P': 4, 'S': 5, 'SA': 5, 'CL': 6,
              'BR': 7, 'I': 8}
protein_map = {'A': 0, 'C': 0, 'N': 1, 'NA': 1, 'O': 2, 'OA': 2, 'S': 3, 'SA': 3}

contacts_kernel = cp.RawKernel(r'''

extern "C" __global__ void CalculateContacts(const unsigned int nligand,
                                  const float* pdLigand_x,
                                  const float* pdLigand_y,
                                  const float* pdLigand_z,
                                  const int *pdLigand_type,
                                  const int *pdLigandOffset,
                                  const float* pdReceptor_x,
                                  const float* pdReceptor_y,
                                  const float* pdReceptor_z,
                                  const int *pdReceptor_type,
                                  const size_t receptorAtoms,
                                  int* pdFeature,
                                  const float cutoff,
                                  const float binSize,
                                  const int nbins,
                                  const int n_receptor_types,
                                  const int n_ligand_types,
                                  const int max_ligand_atoms)
{
    const float cutoff2 = cutoff * cutoff;
    const int nfeatures = nbins*n_ligand_types*n_receptor_types;

    unsigned int ligandIdx = blockIdx.y+gridDim.y*blockIdx.z;
    if (ligandIdx >= nligand)
        return;

    extern  __shared__ char s_data[];
    float3 *sLigandPos = (float3 *) s_data;
    unsigned int *sFeature = (unsigned int *) (sLigandPos + max_ligand_atoms);
    int *sLigandOffset = (int *) (sFeature + nfeatures);

    // Zero feature map
    for (size_t i = threadIdx.x; i < nfeatures; i += blockDim.x)
    {
        sFeature[i] = 0;
    }

    unsigned int ligandAtoms = pdLigandOffset[ligandIdx+1] - pdLigandOffset[ligandIdx];
    size_t offset = pdLigandOffset[ligandIdx];

    // Read ligand
    for (size_t i = threadIdx.x; i < ligandAtoms; i += blockDim.x)
    {
        sLigandPos[i].x = pdLigand_x[offset+i];
        sLigandPos[i].y = pdLigand_y[offset+i];
        sLigandPos[i].z = pdLigand_z[offset+i];
        int type = pdLigand_type[offset+i];
        if (type >= 0)
        {
            sLigandOffset[i] = type * (n_receptor_types * nbins);
        } else {
            // ignore this type
            sLigandOffset[i] = -1;
        }
    }

    __threadfence();
    __syncthreads();

    // Read protein atom for thread
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < receptorAtoms)
    {
        int type = pdReceptor_type[pos];

        // Calculate contacts
        for (size_t i = 0; i < ligandAtoms; i++)
        {
            float dx = pdReceptor_x[pos] - sLigandPos[i].x;
            float dy = pdReceptor_y[pos] - sLigandPos[i].y;
            float dz = pdReceptor_z[pos] - sLigandPos[i].z;
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < cutoff2 && sLigandOffset[i] >= 0)
            {
                float r = sqrt(r2);
                int bin = r / binSize;
                atomicAdd(&sFeature[type*nbins + sLigandOffset[i] + bin], 1);
            }
        }
    }
    __threadfence();
    __syncthreads();

    // Output final counts
    for (size_t i = threadIdx.x; i < nfeatures; i += blockDim.x)
    {
        if (sFeature[i] > 0)
        {
            atomicAdd(&pdFeature[nfeatures*ligandIdx+i], sFeature[i]);
        }
    }

}''', 'CalculateContacts')

def compute(x_ligand, y_ligand, z_ligand, types_ligand, begin_offsets,
             x_receptor, y_receptor, z_receptor, types_receptor,
             cutoff=12.0, binsize=2.0, nbins=6, n_receptor_types=4,
             n_ligand_types=9, max_ligand_atoms=128):
    block_size = 128
    nblocks = x_receptor.shape[0]//block_size + 1
    nligands = len(begin_offsets)-1
    if nligands == 0:
        return cp.array([])

    nfeatures = n_receptor_types * n_ligand_types * nbins
    shared_bytes = 12*max_ligand_atoms + 4*nfeatures + 4*max_ligand_atoms;

    features = cp.zeros((nligands,nfeatures),dtype=cp.int32)

    grid = [nblocks, nligands,1]
    with cp.cuda.Device() as device:
        if grid[1] > device.attributes['MaxGridDimY']:
            grid[2] = grid[1]//device.attributes['MaxGridDimY']+1
            grid[1] = device.attributes['MaxGridDimY']
    grid = tuple(grid)

    contacts_kernel(grid,(block_size,),(nligands,
                                        x_ligand,
                                        y_ligand,
                                        z_ligand,
                                        types_ligand,
                                        begin_offsets,
                                        x_receptor,
                                        y_receptor,
                                        z_receptor,
                                        types_receptor,
                                        x_receptor.shape[0],
                                        features,
                                        cp.float32(cutoff),
                                        cp.float32(binsize),
                                        nbins,
                                        n_receptor_types,
                                        n_ligand_types,
                                        max_ligand_atoms),
                    shared_mem=shared_bytes)
    return features
