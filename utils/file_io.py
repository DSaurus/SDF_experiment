def export_obj(file_name, verts, faces, normals):
    f = open(file_name, 'w')
    for v in verts:
        f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
    # for n in normals:
    #     f.write("vn %f %f %f\n" % (n[0], n[1], n[2]))
    for face in faces:
        face += 1
        f.write("f %d/%d %d/%d %d/%d\n" % (face[0], face[0], face[2], face[2], face[1], face[1]))


def export_pts_cloud(file_name, pts):
    f = open(file_name, 'w')
    for v in pts:
        f.write("v %f %f %f\n" % (v[0], v[1], v[2]))

def export_color_pts(file_name, pts, sdf):
    f = open(file_name, 'w')
    for i in range(pts.shape[0]):
        v = pts[i]
        if sdf[i] < 0.5:
            f.write("v %f %f %f 255 0 0\n" % (v[0], v[1], v[2]))
        else:
            f.write("v %f %f %f 0 0 255\n" % (v[0], v[1], v[2]))

def export_one_pts(file_name, pts, sdf):
    f = open(file_name, 'w')
    for i in range(pts.shape[0]):
        v = pts[i]
        if sdf[i] > 0.5:
            f.write("v %f %f %f 0 255 0\n" % (v[0], v[1], v[2]))