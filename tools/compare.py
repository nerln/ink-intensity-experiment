import sys, numpy as np, pathlib
def load_zarr_u8(root, shape, chunks):
    root=pathlib.Path(root)
    Z,Y,X=shape; cz,cy,cx=chunks
    out=np.zeros(shape,np.uint8)
    for iy in range((Y+cy-1)//cy):
        for ix in range((X+cx-1)//cx):
            p=root/'0'/f'{iy}'/f'{ix}' if False else root/'0'/'0'/f'{iy}'/f'{ix}'
            if not p.exists(): continue
            c=np.frombuffer(p.read_bytes(),np.uint8).reshape(chunks)
            out[:, iy*cy:min(Y,(iy+1)*cy), ix*cx:min(X,(ix+1)*cx)] = \
                c[:, :min(cy,Y-iy*cy), :min(cx,X-ix*cx)]
    return out
r = load_zarr_u8('renders/render_131838.zarr',(33,512,512),(33,128,128))
ref = np.load('renders/_ref_published_131838_roi.npy')
np.save('renders/_mine_131838_roi.npy', r)
print('mio  min/max/mean %d %d %.3f  zeri=%d'%(r.min(),r.max(),r.mean(),(r==0).sum()))
print('ref  min/max/mean %d %d %.3f  zeri=%d'%(ref.min(),ref.max(),ref.mean(),(ref==0).sum()))
for name,a in (('ordine diretto',r),('ordine z invertito',r[::-1])):
    d=a.astype(np.int16)-ref.astype(np.int16)
    eq=int((d==0).sum())
    print('%-20s uguali=%d/%d (%.4f%%)  maxabs=%d  mean=%.5f  meanabs=%.5f'%(
        name,eq,d.size,100*eq/d.size,int(np.abs(d).max()),float(d.mean()),float(np.abs(d).mean())))
