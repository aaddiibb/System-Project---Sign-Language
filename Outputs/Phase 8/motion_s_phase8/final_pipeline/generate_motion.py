
"""Final Motion-S inference API. Load this file from the Phase-8 output directory."""
_PIPELINE = None
def install_pipeline(text_to_gloss, predict_motion_length, generate_rvq_tokens, decode_motion_s_tokens):
    global _PIPELINE
    _PIPELINE=(text_to_gloss,predict_motion_length,generate_rvq_tokens,decode_motion_s_tokens)
def generate_motion(english_text, return_intermediates=False):
    if _PIPELINE is None:
        raise RuntimeError("Call install_pipeline(...) with the frozen manifest-loaded components first; see README_FINAL.txt")
    text_to_gloss,predict_length,generate_tokens,decode_tokens=_PIPELINE
    gloss=text_to_gloss(english_text); length=int(predict_length([gloss])[0]); tokens=generate_tokens(gloss,length); motion=decode_tokens(tokens)
    result={"text":str(english_text),"predicted_gloss":gloss,"predicted_token_length":length,"rvq_tokens":tokens,"motion":motion}
    return result if return_intermediates else motion
