from torch.utils.data import Dataset, DataLoader
from scipy.stats import iqr
from sklearn.cluster import KMeans

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from evaulation import *
from utils import tools
from sds_model import SdActiveSpeaker

# ids = [ '_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', '2fwni_Kjf2M']
ids = ['_mAfwH6i90E', '20TAGRElvfE', 'B1MAUxpKaV8']


def main():
    speaker_dev = []
    non_speaker_dev = []
    for video_id in ids:
        trainLoader = Train_Loader(video_id=video_id, root_dir=video_id)
        trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in trainLoaded:
            for i in range(len(images)):
                asd = SdActiveSpeaker(images[i])
                sds = asd.model()
                if labels['label'][i] == 'SPEAKING':
                    speaker_dev.append(sds)
                else:
                    non_speaker_dev.append(sds)
    
    speaker_dev = [x for x in speaker_dev if x != None]
    non_speaker_dev = [x for x in non_speaker_dev if x != None]

    sq1 = np.quantile(speaker_dev, 0.25)
    sq3 = np.quantile(speaker_dev, 0.75)

    speaker_iqr = (sq3 - sq1)
    speaker_dev = [dev for dev in speaker_dev if dev > (sq1 - 1.5*speaker_iqr) and dev < (sq3+ 1.5*speaker_iqr)]
    print(speaker_dev)

    nsq1 = np.quantile(non_speaker_dev, 0.25)
    nsq3 = np.quantile(non_speaker_dev, 0.75)
    n_speaker_iqr = (nsq3 - nsq1)
    print(nsq1, nsq3, n_speaker_iqr)
    non_speaker_dev = [dev for dev in non_speaker_dev if dev > (nsq1 - 1.5*n_speaker_iqr) and dev < (nsq3 + 1.5*n_speaker_iqr)]

    print('\nSPEAKERS')
    print('Mean: ', np.mean(speaker_dev))
    print('Median: ', np.quantile(speaker_dev, 0.5))
    print('Min', min(speaker_dev))
    print('Max', max(speaker_dev))

    print('\nNON-SPEAKERS')
    print('Mean: ', np.mean(non_speaker_dev))
    print('Median: ', np.quantile(non_speaker_dev, 0.5))
    print('Min', min(non_speaker_dev))
    print('Max', max(non_speaker_dev))


if __name__ == "__main__":
    main()