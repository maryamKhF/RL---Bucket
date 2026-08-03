from .bucket import Bucket

def make_bucket(tx_id,candidates):
    return Bucket(tx_id,tx_id,candidates)
