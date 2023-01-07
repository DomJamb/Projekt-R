class AdvExample():
    """
    Class for representing an adversarial examples
    """
    def __init__(self, initial_pred, attacked_pred, inital_img, attacked_img):
        """
        Constructor for an adversarial example
        Params:
            initial_pred: initial prediction of model
            attacked_pred: prediction of model on an adversarial example
            initial_img: initial data (28x28 image)
            attacked_img: adversarial data (28x28 image)
        """
        self.inital_pred = initial_pred
        self.attacked_pred = attacked_pred
        self.initial_img = inital_img
        self.attacked_img = attacked_img