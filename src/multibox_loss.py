import keras.backend as KB
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, batch_size, alpha=1.0, negative_ratio=2.0
                 , negatives_for_hard=100.0):
        self.__num_classes = num_classes
        self.__alpha = alpha
        self.__negative_ratio = negative_ratio
        self.__negatives_for_hard = negatives_for_hard
        self.__batch_size = batch_size


    def loss(self, y_true, y_pred):
        locs_true = y_true[:, :, :4]
        confs_true = y_true[:, :, 4:-8]
        pboxes_true = y_true[:, :, -8:]
        locs_pred = y_pred[:, :, :4]
        confs_pred = y_pred[:, :, 4:-8]

        box_num = KB.cast(KB.shape(locs_true)[1], 'float32')

        conf_loss = self.__softmax_loss(confs_true, confs_pred)
        loc_loss = self.__smooth_loss(locs_true, locs_pred)

        positive_flgs = pboxes_true[:, :, 0]
        positive_num = tf.reduce_sum(positive_flgs, axis=-1)
        positive_loc_loss = tf.reduce_sum(loc_loss * positive_flgs, axis=1)
        positive_conf_loss = tf.reduce_sum(conf_loss * positive_flgs, axis=1)
      
        negative_num_batch = self.__calc_negative_num_batch(box_num, positive_num)
        negative_conf_loss = self.__get_negative_conf_loss(box_num, positive_flgs, confs_pred
                                                           , conf_loss, negative_num_batch)
      
        total_loss = self.__get_total_loss(positive_num, positive_loc_loss, positive_conf_loss
                                           , negative_num_batch, negative_conf_loss)
        return total_loss


    def __smooth_loss(self, y_true, y_pred):
        _, shape_a, shape_b = y_pred.get_shape().as_list()
        cons = tf.constant(1.0, shape=(self.__batch_size, shape_a, shape_b))
        diff = (y_true - y_pred)
        diff = tf.where(tf.is_inf(diff), cons, diff)
        abs_loss = tf.abs(diff)
        sq_loss = 0.5 * (diff)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)


    def __softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return -1 * softmax_loss


    def __calc_negative_num_batch(self, box_num, positive_num):
        negative_num = self.__negative_ratio * positive_num
        negative_num = tf.minimum(negative_num, box_num - positive_num)

        masks = tf.greater(negative_num, 0)
        has_min = tf.to_float(tf.reduce_any(masks))
        values = [negative_num, [(1 - has_min) * self.__negatives_for_hard]]
        negative_num = tf.concat(axis=0, values=values)
        boolean_mask = tf.boolean_mask(negative_num, tf.greater(negative_num, 0))

        negative_num_batch = tf.reduce_min(boolean_mask)
        negative_num_batch = KB.cast((negative_num_batch), 'int32')
        return negative_num_batch


    def __get_negative_conf_loss(self, box_num, positive_flgs, confs_pred
                                 , conf_loss, negative_num_batch):
        confs_pred_without_bg_cls = confs_pred[:, :, 1:]
        max_confs = tf.reduce_max(confs_pred_without_bg_cls, axis=2)
        _, ids = tf.nn.top_k(max_confs * (1 - positive_flgs), k=negative_num_batch)
        batch_ids = tf.expand_dims(tf.range(0, self.__batch_size), 1)
        batch_ids = tf.tile(batch_ids, (1, negative_num_batch))
        full_ids = tf.reshape(batch_ids, [-1]) * tf.to_int32(box_num) + tf.reshape(ids, [-1])

        negative_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_ids)
        negative_conf_loss = tf.reshape(negative_conf_loss, [self.__batch_size, negative_num_batch])
        negative_conf_loss = tf.reduce_sum(negative_conf_loss, axis=1)
        return negative_conf_loss


    def __get_total_loss(self, positive_num, positive_loc_loss, positive_conf_loss
                         , negative_num_batch, negative_conf_loss):
        total_loss = positive_conf_loss + negative_conf_loss
        total_loss /= (positive_num + tf.to_float(negative_num_batch))
        positive_num = tf.where(tf.not_equal(positive_num, 0), positive_num, tf.ones_like(positive_num))
        total_loss += (self.__alpha * positive_loc_loss) / positive_num
        return total_loss
