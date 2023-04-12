import base64
import io
from PIL import Image
import pandas as pd
import os
import numpy as np


class MultimodalDF(pd.DataFrame):

    # Constructor that calls the parent constructor and initializes additional arguments
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Custom method to encode a column using base85 encoding
    def encode(self, column_name, inplace=False):
        column = self[column_name]
        encoded_column = column.apply(lambda x: base64.b85encode(x).decode("utf-8") if isinstance(x, bytes) else x)
        if inplace:
            self[column_name] = encoded_column
        return encoded_column

    def decode(self, column_name, inplace=False):
        column = self[column_name]
        decoded_column = column.apply(lambda x: base64.b85decode(x) if isinstance(x, str) else x)
        if inplace:
            self[column_name] = decoded_column
        return decoded_column

    # Custom method to load images from a directory or from encoded data and add them as a new column
    def load_images(self, image_col, image_dir, encode=False):
        if image_dir is None:
            raise ValueError("Image directory must be specified.")
        image_data = []
        for file_name in self[image_col]:
            image = None
            if isinstance(file_name, str) and file_name.endswith(
                    ('.jpg', '.jpeg', '.png', '.gif')):  # If data is a file name
                path = os.path.join(image_dir, file_name)
                # image = Image.open(path)
                with open(path, 'rb') as image_obj:
                    image = image_obj.read()
            image_data.append(image)

        self[image_col] = image_data
        if encode:
            self[image_col] = self.encode(column_name=image_col)
        return self[image_col]

    # Custom method to save encoded images as PNG files in the image directory
    def save_images(self, image_data_col, image_dir, filename_col=None):
        if image_dir is None:
            raise ValueError("Image directory must be specified.")
        # Create the image directory if it does not exist
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        num_zeros = int(np.ceil(np.log10(max(self.index)) + 1))
        file_names = []
        for i, data in enumerate(self[image_data_col]):
            if pd.notna(data) and isinstance(base64.b85decode(data), bytes):  # If data is encoded image data
                filename = '{:0{}}.png'.format(self.index[i], num_zeros)  # Use DataFrame index as file name
                file_names.append(filename)
                path = os.path.join(image_dir, filename)
                decoded_data = base64.b85decode(data)
                image = Image.open(io.BytesIO(decoded_data))
                image.save(path)
        if filename_col is None:
            filename_col = image_data_col
        self[filename_col] = file_names


if __name__ == '__main__':
    df = pd.read_csv('temp/test.csv')
    df = MultimodalDF(df)
    image_data = df.load_images(image_col='Images', image_dir='temp')
    image_encoded_data = df.encode(column_name='Images', inplace=True)
    image_decoded_data = df.decode(column_name='Images')
    df.save_images(image_data_col='Images', image_dir='temp/images2', filename_col='filename_col')
    print(df.head())
