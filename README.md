# Transfer-Learning-for-seismic-inversion
Transfer learning: leverage void location to enhance seismic inverion

Versions: 

        Python 3.9.18
        numpy 1.21.2
        Tensorflow 2.6.0
        Keras 2.2.5
        matplotlib 3.4.2

**Please report confusions/erros to y230z012@ku.edu

The technique is decribed in the paper: 


------------------------------------------------------------------------------------------------------

Data structure:

Data, in the paper, can be downloaded with link and unzip under ~/Data: 
        
        https://kansas-my.sharepoint.com/:f:/g/personal/y230z012_home_ku_edu/Eo0ARJCvWmNAlsZv3L1ssLoBQwKnyjGcDAKwc-8MuhIcgQ?e=A4gclP
There are 16 groups dataset in A11:
        
        each group contains 8192 pairs of seismic shot gathers and velocity models.
        A11v1 to A11v7 and A11v10 to A11v16 are training dataset.
        A11v8 and A11v9 are validation dataset.
        There is a fault in the models of A11v1 to A11v8.
        There is not fault in the models of A11v9 to A11v16.
        The seismic shot gathers are at Data/A11v1/bin_A11v1_Nm8192_Nr72_Nt301_Ns3/Vz.bin, and the shape of array is (8192,72,301,3)
                8192 models, 72 receivers, 301 data length (0.6s), and 3 seismic shot gathers.
        The Vs models are at Data/A11v1/models_masks/vs_combined.bin, and the shape of array is (8192, 241, 81, 1)
                8192 models, 241 horizontial length (120.5 m), and 81 vertical length (40.5 m).
        The masks are at Data/A11v1/models_masks/mask_combined_smth.bin, and the shape of array is same with Vs model.

There are 2 groups in A12:
        
        The Vs model in A12v1 contains a fault and there is no fault in A12v2.
        There are 3 subgroups in A12v1. The Vs are same for each subgroup but with different void number.
        The seismic waveforms are at bin_A12v1_XX_Nm8192_Nr72_Nt301_Ns3
                XX represents void number
                The shape of array is (8192, 72, 301, 3)
        The masks are at Data\A12v1_8192mod_no_Fault_FS72recs_T0.6_SmthMask\Rand_Models_Masks_smthMask\mask_XX\000000\mask_smth
                XX represents void number
                000000 is model number
                the shape of array is (1, 241, 81, 1)
        The velocity models are at Data\A12v1_8192mod_no_Fault_FS72recs_T0.6_SmthMask\Rand_Models_Masks_smthMask\models_with_void_XX\000000\
                XX and 000000 are same with mask
                the shape of array is (1, 241, 81, 1)
        
------------------------------------------------------------------------------------------------------        

CNN models:

Models, in the paper, can be downloaded from the link and unzip under ~/models:

        https://kansas-my.sharepoint.com/:f:/g/personal/y230z012_home_ku_edu/Eo0ARJCvWmNAlsZv3L1ssLoBQwKnyjGcDAKwc-8MuhIcgQ?e=A4gclP
The new trained models are saved under ~/train or ~/transfer

------------------------------------------------------------------------------------------------------  

Training and Testing:

Training and transfer learning program can be executed under ~/train and ~/transfer, respectively. The models are saved under same folder.

Testing programs are under ~/test and ~/transfer_test. The program will automatically generate Figure 5 and 10.








                
            



