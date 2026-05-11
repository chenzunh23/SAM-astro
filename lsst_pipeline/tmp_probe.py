import numpy as np
from astropy.io import fits
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

#cat_path = "~/2026-05-01_scarlet_deblend_demo/catalog/meas-HSC-I-9813-4,5.fits"
cat_path = "output/sam_denoised_32_meas/measure/HSC-I/deepCoadd_meas.fits"
exp_path = "output/sam_denoised_32_meas/detect/HSC-I/deepCoadd_calexp.fits"

with fits.open(cat_path) as hdul:
      cols = hdul[1].columns
      idx = cols.names.index("base_PsfFlux_instFlux") + 1
      print("TFORM:", hdul[1].header[f"TFORM{idx}"])
      print("TUNIT:", hdul[1].header.get(f"TUNIT{idx}"))
      print("TTYPE:", hdul[1].header[f"TTYPE{idx}"])

exp = afwImage.ExposureF(exp_path)
photo_calib = exp.getPhotoCalib()
print("PhotoCalib mean:", photo_calib.getCalibrationMean())
print("instFlux at mag 0:", photo_calib.getInstFluxAtZeroMagnitude())

cat = afwTable.SourceCatalog.readFits(cat_path)
for rec in cat:
      flux = rec.get("base_PsfFlux_instFlux")
      if np.isfinite(flux) and flux > 0:
          mag_by_formula = 31.4 - 2.5 * np.log10(flux)
          mag_by_photocalib = photo_calib.instFluxToMagnitude(flux)
          print("flux:", flux)
          print("mag 31.4 formula:", mag_by_formula)
          print("mag PhotoCalib:", mag_by_photocalib)
          break
