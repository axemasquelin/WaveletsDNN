# WaveletsDNN

### Abstract:

The adoption of low-dose computed tomography (LDCT) as the standard of care for lung cancer screening results in decreased mortality rates in high-risk population while increasing false-positive rate. Convolutional neural networks provide an ideal opportunity to improve malignant nodule detection; however, due to the lack of large adjudicated medical datasets these networks suffer from poor generalizability and overfitting. Using computed tomography images of the thorax from the National Lung Screening Trial (NLST), we compared discrete wavelet transforms (DWTs) against convolutional layers found in a CNN in order to evaluate their ability to classify suspicious lung nodules as either malignant or benign. We explored the use of the DWT as an alternative to the convolutional operations within CNNs in order to decrease the number of parameters to be estimated during training and reduce the risk of overfitting. We found that multi-level DWT performed better than convolutional layers when multiple kernel resolutions were utilized, yielding areas under the receiver-operating curve (AUC) of 94% and 92%, respectively. Furthermore, we found that multi-level DWT reduced the number of network parameters requiring evaluation when compared to a CNN and had a substantially faster convergence rate. We conclude that utilizing multi-level DWT composition in place of early convolutional layers within a DNN may improve for image classification in data-limited domains.

Keywords: Area under the AUC curve; Convolutional neural network; Learning rate; Lung cancer detection. 

### Future Directions and Similar Projects:
