from .dualenc_v4 import SpectroformerIB as DualEncoderEpsNetwork_v4

def get_model(config):
    return DualEncoderEpsNetwork_v4(config, training_phase='finetune')