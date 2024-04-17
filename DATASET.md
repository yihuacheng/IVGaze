![](./assets/logo.png)

## In-Vehicle Gaze Estimation DataSet
We provide an in-vehicle gaze estimation dataset IVGaze. 

- IVGaze contains 44,705 images of 125 subjects. We divide the dataset into three subsets based on subjects. The image numbers of the three subsets are 15,165, 14,674, and 14,866. 
Three-fold cross-validation should be performed on the dataset.

- The dataset was collected between 9 am and 7 pm in outdoor environments, covering a wide range of lighting conditions. 

- We consider two face accessories during the collection: glasses and masks. We also required a few subjects to wear sunglasses to facilitate future research.

## Dataset Structure
```
IVGazeDataset
├── class.label
├── Norm
│   ├── 20220811
│   │   ├── subject0000_out_eye_mask
│   │   │    ├── 1.jpg
│   │   │    ├── ...
│   │   │    ├── ...
│   │   │    ├── 81.jpg
│   │   ├── ...
│   │   ├── ...
│   │   ├── subject0000_out_eye_nomask
│   ├── 20221009
│   ├── 20221010
│   ├── 20221011
│   ├── 20221012
│   ├── 20221013
│   ├── 20221014
│   ├── 20221017
│   ├── 20221018
│   ├── 20221019
│   ├── 20221020
│   └── label_class
│       ├── train1.txt
│       ├── train2.txt
│       └── train3.txt
└── Origin
    ├── 20220811
    ├── ...
    ├── ...
    ├── 20221020
    └── label_class
        ├── train1.txt
        ├── train2.txt
        └── train3.txt
```

- `class.label`: This section offers gaze zone classification details. The first row denotes the class number according to `label_class`. The second row represents the original numbers assigned during the data collection phase. The third row indicates coarse region numbers.
- `Norm`: This section contains normalized images and their corresponding labels.
- `Norm/label_class`: Here, you'll find label files for three-fold validation.
- `Origin`: This section provides original images directly cropped from facial images, along with their label files.

## Usage

To retrieve data from the IVGaze Dataset, begin by reading the label file, such as `Norm/label_class/train1.txt`. Each line in the label file is formatted with space-separated values. You can read one line at a time for processing.

```
root = 'IVGazeDataset/Norm'
with open(os.path.join(root, 'label_class/train1.txt')) as infile:
    lines = infile.readlines()

for line in lines:
    line = line.strip().split(' ')

    # Read the image
    image_name = line[0]
    image = cv2.imread(os.path.join(root, image_name))

    # GT for gaze and zone
    gaze = np.fromstring(line[1], sep=',')
    zone = int(line[3])
```

## Download
To obtain access to the dataset, please send an email to `y.cheng.2@bham.ac.uk`. 
You will receive a Google Drive link within three days for downloading the dataset.

Here's the email prompt for requesting access to the IVGaze Dataset. Please do not change the email subject.

```
Subject: Request for Access to IVGaze Dataset

Dear Yihua,

I hope this email finds you well.

I am writing to request access to the IVGaze Dataset. My name is [Your Name], and I am a [student/researcher] from [Your Affiliation].

I assure you that I will only utilize the dataset for academic and research purposes and will not use it for commercial activities.

Thank you for considering my request. I look forward to receiving access to the dataset.

Best regards,
[Your Name]
```

