import cupy as cp
import cudf

ligand_map = {'A': 0, 'C': 0, 'N': 1, 'NA': 1, 'O': 2, 'OA': 2, 'F': 3, 'P': 4, 'S': 5, 'SA': 5, 'CL': 6,
              'BR': 7, 'I': 8}
protein_map = {'A': 0, 'C': 0, 'N': 1, 'NA': 1, 'O': 2, 'OA': 2, 'S': 3, 'SA': 3}

contacts_kernel = cp.RawKernel(r'''

extern "C" __global__ void CalculateContacts(const float* pdLigand_x,
                                  const float* pdLigand_y,
                                  const float* pdLigand_z,
                                  const int *pdLigand_type,
                                  const int *pdLigandOffset,
                                  const float* pdReceptor_x,
                                  const float* pdReceptor_y,
                                  const float* pdReceptor_z,
                                  const int *pdReceptor_type,
                                  const size_t receptorAtoms,
                                  int* pdFeature)
{
    const int LIGAND_ATOM_TYPES = 9;
    const int RECEPTOR_ATOM_TYPES = 4;
    const int MAX_LIGAND_ATOMS = 128;
    const float cutoff = 12.0f;
    const float binSize = 2.0f;
    const float cutoff2 = cutoff * cutoff;
    const int BINS = 6;
    const int FEATURES = BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES;

    __shared__ float3 sLigandPos[MAX_LIGAND_ATOMS];
    __shared__ unsigned int sFeature[FEATURES];
    __shared__ int sLigandOffset[MAX_LIGAND_ATOMS];

    // Zero feature map
    for (size_t i = threadIdx.x; i < FEATURES; i += blockDim.x)
    {
        sFeature[i] = 0;
    }

    unsigned int ligandIdx = blockIdx.y;
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
            sLigandOffset[i] = type * (RECEPTOR_ATOM_TYPES * BINS);
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
                atomicAdd(&sFeature[type*BINS + sLigandOffset[i] + bin], 1);
            }
        }
    }
    __threadfence();
    __syncthreads();

    // Output final counts
    for (size_t i = threadIdx.x; i < FEATURES; i += blockDim.x)
    {
        if (sFeature[i] > 0)
        {
            atomicAdd(&pdFeature[FEATURES*ligandIdx+i], sFeature[i]);
        }
    }

}''', 'CalculateContacts')

nfeatures = 9*4*6

def compute(x_ligand, y_ligand, z_ligand, types_ligand, begin_offsets,
             x_receptor, y_receptor, z_receptor, types_receptor):
    block_size = 128
    nblocks = x_receptor.shape[0]//block_size + 1
    nligands = len(begin_offsets)-1
    if nligands == 0:
        return cp.array([])
    features = cp.zeros((nligands,nfeatures),dtype=cp.int32)

    contacts_kernel((nblocks,nligands,),(block_size,),(x_ligand,
                                          y_ligand,
                                          z_ligand,
                                          types_ligand,
                                          begin_offsets,
                                          x_receptor,
                                          y_receptor,
                                          z_receptor,
                                          types_receptor,
                                          x_receptor.shape[0],
                                          features))
    return features
