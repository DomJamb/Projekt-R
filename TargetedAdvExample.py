class TargetedAdvExample():
    """
    Class for representing a targeted adversarial example
    """
    def __init__(self, inital_img, attacked_img):
        """
        Constructor for an adversarial example
        Params:
            initial_img: initial data (28x28 image)
            attacked_img: adversarial data (28x28 image)
        """
        self.initial_img = inital_img
        self.attacked_img = attacked_img