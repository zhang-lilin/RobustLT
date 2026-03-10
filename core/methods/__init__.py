from .base_method import *
from .enhance_method import *


def get_method(method, logger, **kwargs):
    base_method, enhance_method = None, None
    for k in BASE_METHOD_DICT:
        if k == method:
            logger and logger.log(f'--- adversarial training: {k}')
            return get_base_method(k, **kwargs)
        elif k in method:
            base_method = k
            break
    assert base_method is not None, 'Base method not found'
    for k in ENHANCE_METHOD_DICT:
        if k in method:
            enhance_method = k
            break
    assert enhance_method is not None, 'Enhance method not found'
    logger and logger.log(f'--- adversarial training {base_method} enhanced by {enhance_method}')
    return get_enhance_method(base_method, enhance_method, **kwargs)


def get_base_method(base_method, **kwargs):
    return BASE_METHOD_DICT[base_method](**kwargs)

def get_enhance_method(base_method, enhance_method, **kwargs):
    return ENHANCE_METHOD_DICT[enhance_method](base_method=base_method, **kwargs)