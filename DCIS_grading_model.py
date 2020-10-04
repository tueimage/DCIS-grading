from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras import backend as K
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.densenet import DenseNet121

#Crossentrop ordinal loss function
def loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)


#Weighted quadratic kappa loss (for validation)
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-10, N=3, bsize=6, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom / (denom + eps)

def DCIS_model(IMAGE_SIZE=512):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    pretrained = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    x = pretrained.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output1 = Dense(3, activation='softmax', name='output_main')(x)
    output2 = Dense(3, activation='softmax', name='output_conf')(x)

    model = Model(pretrained.input, [output1, output2])
    model.compile(SGD(lr=0.0001, momentum=0.95), loss={'output_main': loss, 'output_conf': loss}, metrics={'output_main': kappa_loss, 'output_conf': 'accuracy'})
    return model



