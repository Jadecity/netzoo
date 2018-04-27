import common.dataset as dt
import tensorflow as tf
import tensorlayer as tl

if __name__ == '__main__':
    dataset_path = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'
    dataset = dt.DataSet(dataset_path)
    img_batch, size_batch, labels_batch, bboxes_batch = dataset.getNext()

    with tf.Session() as ss:
        img_batch, size_batch, labels_batch, bboxes_batch = ss.run((img_batch, size_batch, labels_batch, bboxes_batch))
        print(bboxes_batch)
        # tl.visualize.frame(img_batch)