import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib
# image_count = len(list(data_dir.glob('*/*.jpg')))

def load_test_data(data_dir, img_height, img_width, batch_size):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    return test_ds

def test_model(model, test_ds):
    print(test_ds)
    loss, accuracy = model.evaluate(test_ds)
    print("Test Accuracy:", accuracy)
    print("Test Loss:", loss)
            
        

def main():
    
    test_data_dir = '/Users/yeshwanth/Documents/DL/project1/part-2/test_img'
    model_file = 'dlTrained_model.h5'
    # Image dimensions and batch size
    img_height = 180
    img_width = 180
    batch_size = 32
    
    # Load the test dataset
    test_ds = load_test_data(test_data_dir, img_height, img_width, batch_size)
    # Load the trained model
    loaded_model = tf.keras.models.load_model(model_file)
    # Test the model
    test_model(loaded_model, test_ds)

if __name__ == "__main__":
    main()
