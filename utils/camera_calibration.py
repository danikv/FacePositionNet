from math import cos, atan2, asin, degrees
import numpy as np
import cv2

def estimate_camera(model3D, fidu_XY, pose_db_on=False):
    if pose_db_on:
        rmat, tvec = calib_camera(model3D, fidu_XY, pose_db_on=True)
        tvec = tvec.reshape(3,1)
    else:
        rmat, tvec = calib_camera(model3D, fidu_XY)
    RT = np.hstack((rmat, tvec))
    projection_matrix = model3D.out_A * RT
    return projection_matrix, model3D.out_A, rmat, tvec

def calib_camera(model3D, fidu_XY, pose_db_on=False):
    #compute pose using refrence 3D points + query 2D point
    ## np.arange(68)+1 since matlab starts from 1
    if pose_db_on:
        rvecs = fidu_XY[0:3]
        tvec = fidu_XY[3:6]
    else:
        goodind = np.setdiff1d(np.arange(68)+1, model3D.indbad)
        goodind=goodind-1
        print(goodind)
        print(fidu_XY.shape)
        fidu_XY = fidu_XY[goodind,:]
        ret, rvecs, tvec = cv2.solvePnP(model3D.model_TD, fidu_XY, model3D.out_A, None, None, None, False)
    rmat, jacobian = cv2.Rodrigues(rvecs, None)

    inside = calc_inside(model3D.out_A, rmat, tvec, model3D.size_U[1], model3D.size_U[0], model3D.model_TD)
    if(inside == 0):
        tvec = -tvec
        t = np.pi
        RRz180 = np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
        rmat = RRz180*rmat
    return rmat, tvec

def get_yaw(rmat):
    modelview = rmat
    modelview = np.zeros( (3,4 ))
    modelview[0:3,0:3] = rmat.transpose()
    modelview = modelview.reshape(12)
    # Code converted from function: getEulerFromRot()
    angle_y = -asin( modelview[2] )  # Calculate Y-axis angle
    C = cos( angle_y)
    angle_y = degrees(angle_y)

    if np.absolute(C) > 0.005: # Gimball lock?
        trX = modelview[10] / C # No, so get X-axis angle
        trY = -modelview[6] / C
        angle_x = degrees( atan2( trY, trX ) )

        trX = modelview[0] / C  # Get z-axis angle
        trY = - modelview[1] / C
        angle_z = degrees(  atan2( trY, trX) )
    else:
        # Gimball lock has occured
        angle_x = 0
        trX = modelview[5]
        trY = modelview[4]
        angle_z = degrees(  atan2( trY, trX) )

    # Adjust to current mesh setting
    angle_x = 180 - angle_x
    angle_y = angle_y
    angle_z = -angle_z

    out_pitch = angle_x
    out_yaw = angle_y
    out_roll = angle_z

    return out_yaw


def get_opengl_matrices(camera_matrix, rmat, tvec, width, height):
    projection_matrix = np.asmatrix(np.zeros((4,4)))
    near_plane = 0.0001
    far_plane = 10000

    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    px = camera_matrix[0,2]
    py = camera_matrix[1,2]

    projection_matrix[0, 0] = 2.0 * fx / width
    projection_matrix[1, 1] = 2.0 * fy / height
    projection_matrix[0, 2] = 2.0 * (px / width) - 1.0
    projection_matrix[1, 2] = 2.0 * (py / height) - 1.0
    projection_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    projection_matrix[3, 2] = -1
    projection_matrix[2, 3] = -2.0 * far_plane * near_plane / (far_plane - near_plane)

    deg = 180
    t = deg*np.pi/180.
    RRz=np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
    RRy=np.asmatrix([np.cos(t), 0, np.sin(t), 0, 1, 0, -np.sin(t), 0, np.cos(t)]).reshape((3, 3))
    rmat=RRz*RRy*rmat

    mv = np.asmatrix(np.zeros((4, 4)))
    mv[0:3, 0:3] = rmat
    mv[0, 3] = tvec[0]
    mv[1, 3] = -tvec[1]
    mv[2, 3] = -tvec[2]
    mv[3, 3] = 1.
    return mv, projection_matrix


def extract_frustum(camera_matrix, rmat, tvec, width, height):
    mv, proj = get_opengl_matrices(camera_matrix, rmat, tvec, width, height)
    clip = proj * mv
    frustum = np.asmatrix(np.zeros((6 ,4)))
    #/* Extract the numbers for the RIGHT plane */
    frustum[0, :] = clip[3, :] - clip[0, :]
    #/* Normalize the result */
    v = frustum[0, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[0, :] = frustum[0, :]/t

    #/* Extract the numbers for the LEFT plane */
    frustum[1, :] = clip[3, :] + clip[0, :]
    #/* Normalize the result */
    v = frustum[1, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[1, :] = frustum[1, :]/t

    #/* Extract the BOTTOM plane */
    frustum[2, :] = clip[3, :] + clip[1, :]
    #/* Normalize the result */
    v = frustum[2, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[2, :] = frustum[2, :]/t

    #/* Extract the TOP plane */
    frustum[3, :] = clip[3, :] - clip[1, :]
    #/* Normalize the result */
    v = frustum[3, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[3, :] = frustum[3, :]/t

    #/* Extract the FAR plane */
    frustum[4, :] = clip[3, :] - clip[2, :]
    #/* Normalize the result */
    v = frustum[4, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[4, :] = frustum[4, :]/t

    #/* Extract the NEAR plane */
    frustum[5, :] = clip[3, :] + clip[2, :]
    #/* Normalize the result */
    v = frustum[5, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[5, :] = frustum[5, :]/t
    return frustum


def calc_inside(camera_matrix, rmat, tvec, width, height, obj_points):
    frustum = extract_frustum(camera_matrix, rmat, tvec, width, height)
    inside = 0
    for point in obj_points:
        if(point_in_frustum(point[0], point[1], point[2], frustum) > 0):
            inside += 1
    return inside


def point_in_frustum(x, y, z, frustum):
    for p in range(0, 3):
        if(frustum[p, 0] * x + frustum[p, 1] * y + frustum[p, 2] + z + frustum[p, 3] <= 0):
            return False
    return True

def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = -asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))


    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t3d: (3,). 3d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d