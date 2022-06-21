from monai.transforms import Compose, RandSpatialCrop, RandRotate, \
    RandFlip, RandZoom, NormalizeIntensity, RandScaleIntensity, \
    RandAdjustContrast, RandHistogramShift, RandGibbsNoise, \
    RandKSpaceSpikeNoise, HistogramNormalize, RandBiasField

def get_train_transforms():
    transforms_train = Compose([
        HistogramNormalize(),
        NormalizeIntensity(),

        RandScaleIntensity(
            factors=(0.95,1.05),
            prob=0.5
        ),
        RandAdjustContrast(
            prob=0.5,
            gamma=(0.9,1.1)
        ),
        RandHistogramShift(
            prob=0.2
        ),
        RandGibbsNoise(
            prob=0.1,
            alpha=(0,0.5)
        ),
        RandKSpaceSpikeNoise(
            prob=0.1,
            intensity_range=(0.95,1.05)
        ),
        RandBiasField(
            degree=3,
            coeff_range=(0,0.3),
            prob=0.1
        ),
        RandFlip(
            prob=0.3
        ),
        RandRotate(
            range_x=0.2, 
            range_y=0.2, 
            range_z=0.2, 
            prob=0.3, 
            padding_mode='zeros'
        ),
        RandZoom(
            prob=0.3, 
            min_zoom=0.9,
            max_zoom=1.1
        ),
        RandSpatialCrop(
            roi_size=(150,150, 150), 
            random_center=True, 
            random_size=False)
    ])
    return transforms_train

def get_val_transforms():
    transforms_val = Compose([
        HistogramNormalize(),
        NormalizeIntensity(),

        RandSpatialCrop(
            roi_size=(150,150, 150), 
            random_center=True, 
            random_size=False)
    ])
    return transforms_val
    