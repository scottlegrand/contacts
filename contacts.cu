#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <cstdint>

using namespace std;

static const int LIGAND_ATOM_TYPES = 9;
static const int RECEPTOR_ATOM_TYPES = 4;
static const int MAX_LIGAND_ATOMS = 128;
static const float cutoff = 12.0f;
static const float binSize = 2.0f;
static const float cutoff2 = cutoff * cutoff;
static const int BINS = 6;
static const int FEATURES = BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES;


struct Atom
{
    float _x;
    float _y;
    float _z;
    int _type;
};

#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        printf("%s %s\n", s, cudaGetErrorString(status)); \
        assert(0); \
        cudaDeviceReset(); \
        exit(-1); \
    }

#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            assert(0); \
            getGpu().Shutdown(); \
            exit(-1); \
        } \
    }



bool ReadPDBQT(string fname, unordered_map<string, int>& map, vector<Atom>& vMolecule)
{
    vMolecule.resize(0);
    ifstream input(fname);
    Atom a;
    for( std::string line; getline( input, line ); )
    {
        if (line.rfind("ATOM", 0) == 0) 
        {

            char type[16];
            const char* buff = line.c_str();
            sscanf(&buff[77], "%s", type);
            std::unordered_map<std::string,int>::const_iterator got = map.find(type);
            if (got != map.end())
            {          
            
                sscanf(&buff[30], "%8f%8f%8f", &a._x,&a._y,&a._z);
                a._type = got->second;     
                //printf("%8.3f %8.3f %8.3f %3s %d\n", a._x, a._y, a._z, type, a._type);
                vMolecule.push_back(a);
            }
        }
    }
    
    return true;    
}


Atom* UploadPDBQT(vector<Atom>& vMolecule)
{
    Atom* pdMolecule;
    cudaError_t status = cudaMalloc((void**)&pdMolecule, vMolecule.size() * sizeof(Atom));
    RTERROR(status, "UploadPDBQT: Failed to allocate memory for molecule.\n");
    status = cudaMemcpy(pdMolecule, vMolecule.data(), vMolecule.size() * sizeof(Atom), cudaMemcpyDefault);
    RTERROR(status, "UploadPDBQT: Failed to upload molecule.\n");
    return pdMolecule;
}

__global__ void CalculateContacts(const Atom* pdLigand,
                                  const size_t ligandAtoms,
                                  const Atom* pdReceptor,
                                  const size_t receptorAtoms,
                                  uint32_t* pdFeature)
{
__shared__ uint32_t sFeature[BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES];
__shared__ float3 sLigandPos[MAX_LIGAND_ATOMS];
__shared__ int sLigandOffset[MAX_LIGAND_ATOMS];

    // Zero feature map
    for (size_t i = threadIdx.x; i < BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES; i += blockDim.x)
    {
        sFeature[i] = 0;
    }
    
    // Read ligand
    for (size_t i = threadIdx.x; i < ligandAtoms; i += blockDim.x)
    {
        Atom a = pdLigand[i];
        sLigandPos[i].x = a._x;
        sLigandPos[i].y = a._y;
        sLigandPos[i].z = a._z;
        sLigandOffset[i] = a._type * (RECEPTOR_ATOM_TYPES * BINS);
    }
    
    __threadfence();
    __syncthreads();
    
    // Read protein atom for thread
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < receptorAtoms)
    {
        Atom a = pdReceptor[pos];
        a._type *= BINS;
        
        // Calculate contacts
        for (size_t i = 0; i < ligandAtoms; i++)
        {
            float dx = a._x - sLigandPos[i].x;
            float dy = a._y - sLigandPos[i].y;            
            float dz = a._z - sLigandPos[i].z;
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < cutoff2)
            {
                float r = sqrt(r2);
                int bin = r / binSize;
                atomicAdd(&sFeature[a._type + sLigandOffset[i] + bin], 1);
            }        
        }
    }
    __threadfence();
    __syncthreads();
    
    
    // Output final counts
    for (size_t i = threadIdx.x; i < BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES; i += blockDim.x)
    {
        if (sFeature[i] > 0)
        {
            atomicAdd(&pdFeature[i], sFeature[i]);
        }
    }
}
    





int main(int argc, char** argv)
{
    // Initialize atom data
    //                      C  N  O  F   P   Cl  S   Br  I
    //                      0, 1, 2, 3,  4,  5,  6   7, 8
    //ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]

    //                           0, 1, 2, 3
    //    protein_atomic_nums = [6, 7, 8, 16]
    //                          C N O S

    unordered_map<string, int> ligandMap;
    ligandMap["A"]  =  0;
    ligandMap["C"]  =  0;
    ligandMap["N"]  =  1;
    ligandMap["NA"] =  1;
    ligandMap["O"]  =  2;
    ligandMap["OA"] =  2;
    ligandMap["F"]  =  3;   
    ligandMap["P"]  =  4;
    ligandMap["CL"] =  5;
    ligandMap["S"]  =  6;
    ligandMap["SA"]  = 6;
    ligandMap["BR"] =  7;
    ligandMap["I"]  =  8;
    
    unordered_map<string, int> proteinMap;
    proteinMap["A"]  =  0;
    proteinMap["C"]  =  0;
    proteinMap["N"]  =  1;
    proteinMap["NA"] =  1;
    proteinMap["O"]  =  2;
    proteinMap["OA"] =  2;
    proteinMap["S"]  =  3;
    proteinMap["SA"]  = 3;
    
    cudaFree(0);
       
    // Read ligand
    vector<Atom> vLigand;
    ReadPDBQT("test.pdbqt", ligandMap, vLigand);
    Atom* pdLigand = UploadPDBQT(vLigand);
        
    // Read receptor
    vector<Atom> vReceptor;
    ReadPDBQT("final_Mpro_5R84_gast.pdbqt", proteinMap, vReceptor);
    Atom* pdReceptor = UploadPDBQT(vReceptor);
    cout << vLigand.size() << " " << vReceptor.size() << endl;

    
    // Allocate feature vector
    uint32_t* pdFeature;
    cudaError_t status = cudaMalloc((void**)&pdFeature, BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES * sizeof(uint32_t));
    RTERROR(status, "main: Failed to allocate memory for feature vector.\n");
    status = cudaMemset(pdFeature, 0, BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES * sizeof(uint32_t));
    RTERROR(status, "main: Failed to zero feature vector.\n");
    
    // Calculate contacts
    uint32_t blockSize = 128;
    uint32_t blocks = (uint32_t)((vReceptor.size() + blockSize - 1) / blockSize);
    CalculateContacts<<<blocks, blockSize>>>(pdLigand, vLigand.size(), pdReceptor, vReceptor.size(), pdFeature);
    
    // Download contacts
    vector<uint32_t> vFeature(BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES);
    status = cudaMemcpy(vFeature.data(), pdFeature, BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES * sizeof(uint32_t), cudaMemcpyDefault);
    
    
    // Print result
    for (size_t i = 0; i < BINS * LIGAND_ATOM_TYPES * RECEPTOR_ATOM_TYPES; i++)
        printf("%3lu %6u\n", i, vFeature[i]);
 
    status = cudaFree(pdFeature);       
    RTERROR(status, "main: Failed to deallocate memory for feature vector.\n");
    status = cudaFree(pdLigand);
    RTERROR(status, "main: Failed to deallocate memory for ligand.\n");    
    status = cudaFree(pdReceptor);
    RTERROR(status, "main: Failed to deallocate memory for receptor.\n");
    return 0;
}
