import lasagne

class lasagne_embedding_layer_2(lasagne.layers.Layer):
    def __init__(self, incoming, output_size, W, **kwargs):

        super(lasagne_embedding_layer_2, self).__init__(incoming, **kwargs)
        self.output_size = output_size
        self.W = W

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]