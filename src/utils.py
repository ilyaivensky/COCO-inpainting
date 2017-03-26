import matplotlib.pyplot as plt
import numpy as np

def iterate_minibatches(hfp, split, batchsize, shuffle=False):
        
    x = hfp['/%s/x' % split]
    z = hfp['/%s/z' % split]
    y = hfp['/%s/y' % split]
    
    assert len(x) == len(y)
    assert len(x) == len(z)

    if shuffle:
        indices = np.arange(len(x))
        np.random.shuffle(indices)
    for start_idx in range(0, len(x) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield x[excerpt], z[excerpt], y[excerpt]
            
   
        
def show_samples(target, samples):

    nb_samples = len(samples)
    s = (samples.transpose(0,2,3,1) * 255).astype(np.uint8)
    
    for i in range(nb_samples):
            
        plt.subplot(2, nb_samples, i+1)
        plt.imshow((target[i] * 255).astype(np.uint8))
    
        img_pred = np.copy(s[i])
        plt.subplot(2, nb_samples, nb_samples+i+1)
        plt.imshow(img_pred)
    
    plt.show()