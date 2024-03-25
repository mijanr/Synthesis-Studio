import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_features, gen_features):
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':
    # test FID calculation
    real_features = np.random.rand(1000, 10)
    gen_features = np.random.rand(1000, 10)
    fid = calculate_fid(real_features, gen_features)
    print('FID: %.3f' % fid)