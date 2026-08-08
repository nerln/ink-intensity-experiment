import numpy as np, pathlib, json
def read(root):
    root=pathlib.Path(root); m=json.loads((root/'0'/'.zarray').read_text())
    assert m['compressor'] is None
    Z,Y,X=m['shape']; cz,cy,cx=m['chunks']; dt=np.dtype(m['dtype'])
    out=np.zeros((Z,Y,X),dt)
    for iy in range((Y+cy-1)//cy):
        for ix in range((X+cx-1)//cx):
            p=root/'0'/'0'/str(iy)/str(ix)
            if not p.exists(): continue
            c=np.frombuffer(p.read_bytes(),dt).reshape((cz,cy,cx))
            out[:,iy*cy:min(Y,(iy+1)*cy),ix*cx:min(X,(ix+1)*cx)]=c[:,:min(cy,Y-iy*cy),:min(cx,X-ix*cx)]
    return out
