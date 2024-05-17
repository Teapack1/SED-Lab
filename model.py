import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Reshape,
    Flatten,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.convnext import ConvNeXtBase, ConvNeXtTiny


class Deep_NN:
    def __init__(self, dim1, dim2, dim3, num_classes):
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.num_classes = num_classes
        

    def defaultCNN(self, type="default"):
        
        input_shape = (self.dim1, self.dim2, self.dim3)
        
        print(f"Active Neural Net: 'defaultCNN' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")
        
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        return model

    def customCNN1(self, type="default"):
        input_shape = (self.dim1, self.dim2, self.dim3)

        print(f"Active Neural Net: 'customCNN1' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")

        model = Sequential(
            [
                # Convolutuonal block 1
                Conv2D(
                    16,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=input_shape,
                ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Convolutuonal block 2
                Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Convolutuonal block 3
                Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Output
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        return model

    def mobilenetv3_nn(self, type="small"):
        input_shape = (
            self.dim1,
            self.dim2,
            self.dim3,
        )  # Make sure this shape includes the channels dimension
        
        print(f"Active Neural Net: 'mobilenetv3_nn' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")
        print("Expected data range: 0 - 255")
        
        if type == "large":
            base_model = MobileNetV3Large(
                weights=None,  # No pre-trained weights
                include_top=True,  # Include the top (classification) layer
                input_shape=input_shape,
                classes=self.num_classes,  # Specify the number of classes
            )
        elif type == "small":
            base_model = MobileNetV3Small(
                weights=None,  # No pre-trained weights
                include_top=True,  # Include the top (classification) layer
                input_shape=input_shape,
                classes=self.num_classes,  # Specify the number of classes
            )
            
        for layer in base_model.layers:
            layer.trainable = True

        # The base model already includes a `Dense` layer with `self.num_classes` units
        model = Model(inputs=base_model.input, outputs=base_model.output)

        return model

    def effnetv2_nn(self, type="default"):
        input_shape = (
            self.dim1,
            self.dim2,
            self.dim3,
        )  # Make sure this shape includes the channels dimension
        
        print(f"Active Neural Net: 'effnetv2_nn' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")
        print("Expected data range: 0 - 255")
        
        base_model = EfficientNetV2S(
            weights=None,  # No pre-trained weights
            include_top=True,  # Include the top (classification) layer
            input_shape=input_shape,
            classes=self.num_classes,  # Specify the number of classes
        )

        for layer in base_model.layers:
            layer.trainable = True

        # The base model already includes a `Dense` layer with `self.num_classes` units
        model = Model(inputs=base_model.input, outputs=base_model.output)

        return model
    

    def convnext_nn(self, type="base"):
        
        input_shape = (
            self.dim1,
            self.dim2,
            self.dim3,
        )  # Make sure this shape includes the channels dimension

        print(f"Active Neural Net: 'convnext_nn' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")
        print("Expected data range: 0 - 255")
        
        if type == "base":
            base_model = ConvNeXtBase(
                weights=None,  # No pre-trained weights
                include_top=True,  # Include the top (classification) layer
                input_shape=input_shape,
                classes=self.num_classes,  # Specify the number of classes
            )
        elif type == "tiny":
            base_model = ConvNeXtTiny(
                weights=None,  # No pre-trained weights
                include_top=True,  # Include the top (classification) layer
                input_shape=input_shape,
                classes=self.num_classes,  # Specify the number of classes
            )
        else:
            raise ValueError("Type must be either 'base' or 'tiny'.")

        for layer in base_model.layers:
            layer.trainable = True

        # The base model already includes a `Dense` layer with `self.num_classes` units
        model = Model(inputs=base_model.input, outputs=base_model.output)

        return model



    def SmallerVGGNet(self, type="default"):
        input_shape = (self.dim1, self.dim2, self.dim3)
        chanDim = -1

        print(f"Active Neural Net: 'SmallerVGGNet' {type}, \n Input Shape: {input_shape} \n Num Classes: {self.num_classes} \n")

        model = Sequential(
            [
                # Convolutional block 1
                Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=input_shape,
                ),
                BatchNormalization(axis=chanDim),
                MaxPooling2D(pool_size=(3, 3)),
                Dropout(0.25),

                # Convolutional block 2
                Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(axis=chanDim),
                Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(axis=chanDim),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                # Convolutional block 3
                Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(axis=chanDim),
                Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
                BatchNormalization(axis=chanDim),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                # Fully connected layer
                Flatten(),
                Dense(1024, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),

                # Output layer
                Dense(self.num_classes, activation="softmax"),
            ]
        )
        # Return the constructed network architecture
        return model