# fn="aaa6a05cc_224_0_[2449 2673  661  885]_0.0.npz"
# fn="aaa6a05cc_224_0_[2004 2228 5072 5296]_0.5984335140306123.npz"
# ex1=np.load(fn)['a']
i_ids=["aaa6a05cc","095bf7a1f","2f6ecfcdf","0486052bb","54f2eec69","1e2425f28", "cb2d976f4"]
for im in i_ids:
    try:
        #path=path+str(min_overlap)+"\\"
        path = "D:\\hubmap\\data_npz\\"+im+"\\"
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        
    else:
        print ("Successfully created the directory %s " % path)

    
    getImageAndMask(im,True)
    for ol in [0,50,100]:
        
        mgst=make_grid_save_tiles((image.shape[0],image.shape[1]),  image, im, mask, window=224, min_overlap=ol, save=True, path="D:\\hubmap\\data_npz\\"+im+"\\"+str(ol)+"\\",ow=False)




def make_grid_save_tiles(shape, image, image_name, mask, window=256, min_overlap=32, save=True,path=BASE_PATH, ow=False):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    
    import os.path
    # https://stackabuse.com/creating-and-deleting-directories-with-python/
    try:
        #path=path+str(min_overlap)+"\\"
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        
    else:
        print ("Successfully created the directory %s " % path)
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    cc=0
    for i in range(nx):
        for j in range(ny):
            cc+=1
            print(f"slice #{cc} out of {nx*ny} ({np.round(cc/(nx*ny)*100,2)}%)")
            
            slices[i,j] = x1[i], x2[i], y1[j], y2[j] 
            
            if save:
                mask_per=np.round(np.mean(mask[x1[i]:x2[i], y1[j]:y2[j]]),4)
                fname=f"_{image_name}_{window}_{min_overlap}_{slices[i,j]}_{mask_per}_.npz"
                pj=os.path.join(path,fname)
                
                
                if (os.path.exists(pj) and ow) or (not os.path.exists(pj)):
                    if min_overlap>0:
                        if mask_per>0.0:

                            np.savez_compressed(pj, a=image[x1[i]:x2[i], y1[j]:y2[j]]/255.0)
                            print("saved: ",fname)
                    else:
                        np.savez_compressed(pj, a=image[x1[i]:x2[i], y1[j]:y2[j]]/255.0)

                        print("saved: ",fname)
                else:
                    print(f"file {fname} exists at {path}. not overwriting...")

    return slices.reshape(nx*ny,4)