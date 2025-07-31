data = load('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/rsfMRI_ROISignals_phenotypic_ADNI2_ADNI3_20231210.mat')

data = table2struct(data.fMRIdata_ADNI2_ADNI3)
save('/home/xinxu/Lehigh/Codes/BICLab_data/ADNI/adni23_label.mat', 'data')