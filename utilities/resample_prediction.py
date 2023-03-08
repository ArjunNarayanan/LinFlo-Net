import SimpleITK as sitk

def resample_prediction(self, upsample=False):
    # resample prediction so it matches the original image
    resampled = []
    for pred in [self.prediction, self.prob_prediction]:
        im = sitk.GetImageFromArray(pred.transpose(2, 1, 0))
        im.SetSpacing(self.image_resampled.GetSpacing())
        im.SetOrigin(self.image_resampled.GetOrigin())
        im.SetDirection(self.image_resampled.GetDirection())

        ori_im = self.image_vol
        if upsample:
            size = ori_im.GetSize()
            spacing = ori_im.GetSpacing()
            new_size = [max(s, 128) for s in size]
            ref_im = sitk.Image(new_size, ori_im.GetPixelIDValue())
            ref_im.SetOrigin(ori_im.GetOrigin())
            ref_im.SetDirection(ori_im.GetDirection())
            ref_im.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(new_size, size, spacing)])
            ctr_im = sitk.Resample(ori_im, ref_im, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear)
            resampled.append(centering(im, ctr_im, order=0))
        else:
            resampled.append(centering(im, ori_im, order=0))

    self.prediction = resampled[0]
    self.prob_prediction = resampled[1]

    return [self.prediction, self.prob_prediction]