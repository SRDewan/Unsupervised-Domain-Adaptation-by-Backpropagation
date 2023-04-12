import torch.nn as nn

class GTSRB_CNN_regular(nn.Module):

    def __init__(self):
        super(GTSRB_CNN, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module("f_conv1", nn.Conv2d(3, 96, kernel_size = 5))
        self.feature.add_module("f_bn1", nn.BatchNorm2d(96))
        self.feature.add_module("f_relu1", nn.ReLU(True))
        self.feature.add_module("f_pool1", nn.MaxPool2d(2, 2))

        self.feature.add_module("f_conv2", nn.Conv2d(96, 144, kernel_size = 3))
        self.feature.add_module("f_bn2", nn.BatchNorm2d(144))
        self.feature.add_module("f_relu2", nn.ReLU(True))
        self.feature.add_module("f_pool2", nn.MaxPool2d(2, 2))

        self.feature.add_module("f_conv3", nn.Conv2d(144, 256, kernel_size = 5))
        self.feature.add_module("f_bn3", nn.BatchNorm2d(256))
        self.feature.add_module("f_relu3", nn.ReLU(True))
        self.feature.add_module("f_pool3", nn.MaxPool2d(2, 2))
        # self.feature.add_module("f_drop1", nn.Dropout2d())
        self.feature.add_module("f_flatten", nn.Flatten())


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module("c_fc1", nn.Linear(256 * 3 * 3, 512))
        self.class_classifier.add_module("c_bn1", nn.BatchNorm1d(512))
        self.class_classifier.add_module("c_relu1", nn.ReLU(True))
        # self.class_classifier.add_module("c_drop1", nn.Dropout1d())

        self.class_classifier.add_module("c_fc2", nn.Linear(512, 43))
        # self.class_classifier.add_module("c_bn2", nn.BatchNorm1d(43))
        self.class_classifier.add_module("c_softmax", nn.LogSoftmax())

    def forward(self, input_data):
        feature = self.feature(input_data)
        class_output = self.class_classifier(feature)

        return class_output
