echo -e "\e[32mStarting noisy evaluation\e[0m"
python utils/run_cutout_magnitude_experiment.py     --coadd-root fits/noisy    --reference-catalog /home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/meas-HSC-I-9813-4,5.fits  \
    --output-root output/cutout_magnitude_experiment_grid/gri_64_noisy    --methods lsst sam \
    --lsst-workers 16     --sam-gpus 0,1,2     --sam-workers-per-gpu 2  \
    --bin-size 0.5   --continue-on-error --mag-min 18 --mag-max 30 \
    --sam-extra-args "--sam-points-per-side 64"
echo -e "\e[32m Starting denoised evalutaion\e[0m"
python utils/run_cutout_magnitude_experiment.py     --coadd-root fits/projection_cutout/     --reference-catalog /home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/meas-HSC-I-9813-4,5.fits  \
    --output-root output/cutout_magnitude_experiment_grid/gri_64_denoised --methods lsst sam  \
    --lsst-workers 16     --sam-gpus 0,1,2     --sam-workers-per-gpu 2  \
    --bin-size 0.5   --continue-on-error --mag-min 18 --mag-max 30 \
    --sam-extra-args "--sam-points-per-side 64"
