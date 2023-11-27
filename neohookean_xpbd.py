import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# cube with 5 tetrahedrons 
pos_np = np.array(
    [[-1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0],
     [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]],
    dtype=np.float32)
tet_np = np.array([[0, 1, 2, 5], [0, 2, 5, 7], [0, 4, 5, 7], [2, 3, 0, 7],[2, 5, 6, 7]], dtype=np.int32)
tri_np = np.array([0,1,2,0,2,3,4,5,7,5,6,7,0,5,4,0,1,5,2,3,7,2,7,6,0,4,7,0,7,3,1,2,5,2,6,5], dtype=np.int32)

pos = ti.Vector.field(3, dtype=ti.f32, shape=8)
tet = ti.Vector.field(4, dtype=ti.i32, shape=5)
tri = ti.field(dtype=ti.i32, shape=12 * 3)
V = ti.field(dtype=ti.f32, shape=5)
old_pos = ti.Vector.field(3, dtype=ti.f32, shape=8)
vel = ti.Vector.field(3, dtype=ti.f32, shape=8)
mass = ti.field(dtype=ti.f32, shape=8)
inv_mass = ti.field(dtype=ti.f32, shape=8)

B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=5)

# xpbd compliance parameters
alpha_hydrostatic = ti.field(dtype=ti.f32, shape=5)
alpha_deviatoric = ti.field(dtype=ti.f32, shape=5)

# configuration 
h = 0.01
density = 10.0
maxIte = 10
youngs_modulus, poissons_ratio = 1.0e5, 0.45
lame_lambda = youngs_modulus * poissons_ratio / (1 + poissons_ratio) / (
    1 - 2 * poissons_ratio)
lame_mu = youngs_modulus / 2 / (1 + poissons_ratio)



@ti.kernel
def init(pos_np: ti.types.ndarray(), tet_np: ti.types.ndarray(),
         tri_np: ti.types.ndarray()):
    for i in pos:
        pos[i] = ti.Vector([pos_np[i, 0], pos_np[i, 1], pos_np[i, 2]])
    for i in tet:
        tet[i] = ti.Vector(
            [tet_np[i, 0], tet_np[i, 1], tet_np[i, 2], tet_np[i, 3]])
    for i in tri:
        tri[i] = tri_np[i]


@ti.kernel
def init_phy(density: ti.f32):
    for i in pos:
        old_pos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
    for i in mass:
        mass[i] = 0.0
    for i in tet:
        a, b, c, d = tet[i]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        D_m = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        B[i] = D_m.inverse()
        
        V[i] = 1.0 / 6.0 * ti.abs(D_m.determinant())  # rest volume
        avg_mass = density * V[i] / 4.0
        mass[a] += avg_mass
        mass[b] += avg_mass
        mass[c] += avg_mass
        mass[d] += avg_mass
    for i in pos:
        inv_mass[i] = 1.0 / mass[i]


@ti.kernel
def init_alpha(h: ti.f32):
    inv_h2 = 1.0 / h / h
    inv_lambda = 1.0 / lame_lambda
    inv_mu = 1.0 / lame_mu
    for i in tet:
        alpha_hydrostatic[i] = inv_h2 * inv_lambda * 1.0 / V[i]
        alpha_deviatoric[i] = inv_h2 * inv_mu * 1.0 / V[i]


@ti.kernel
def semi_euler(h: ti.f32):
    for i in pos:
        vel[i] += h * ti.Vector([0.0, -9.8, 0.0])
        old_pos[i] = pos[i]
        pos[i] += h * vel[i]


@ti.kernel
def project_constraints():
    for i in tet:
        print("-----------------project_deviatoric_constraints-------------------")
        a, b, c, d = tet[i]
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        D_s = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        F = D_s @ B[i]
        F_T = F.transpose()
        rs = (F_T @ F).trace()
        # G = 1.0 / rs * F @ B[i].transpose() # not robust when colliding with ground
        G = 2.0 * F @ B[i].transpose()
        g1 = ti.Vector([G[0, 0], G[1, 0], G[2, 0]])
        g2 = ti.Vector([G[0, 1], G[1, 1], G[2, 1]])
        g3 = ti.Vector([G[0, 2], G[1, 2], G[2, 2]])
        g0 = -(g1 + g2 + g3)

        # compute constraints
        constriaint = rs - 3.0
        w0, w1, w2, w3 = inv_mass[a], inv_mass[b], inv_mass[c], inv_mass[d]
        sum_w = w0 * g0.norm_sqr() + w1 * g1.norm_sqr() + w2 * g2.norm_sqr(
        ) + w3 * g3.norm_sqr()
        if (sum_w == 0.0):
            continue
        lambda_ = -constriaint / (sum_w + alpha_deviatoric[i])
        pos[a] += w0 * lambda_ * g0
        pos[b] += w1 * lambda_ * g1
        pos[c] += w2 * lambda_ * g2
        pos[d] += w3 * lambda_ * g3

        print("-----------------project_hydrostatic_constraint-------------------")
        p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        D_s = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
        F = D_s @ B[i]
        f1 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        f2 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        f3 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])
        df0 = f2.cross(f3)
        df1 = f3.cross(f1)
        df2 = f1.cross(f2)

        G = ti.Matrix.cols([df0, df1, df2]) @ B[i].transpose()
        g1 = ti.Vector([G[0, 0], G[1, 0], G[2, 0]])
        g2 = ti.Vector([G[0, 1], G[1, 1], G[2, 1]])
        g3 = ti.Vector([G[0, 2], G[1, 2], G[2, 2]])
        g0 = -(g1 + g2 + g3)
        # compute constraints
        constriaint = F.determinant() - 1.0

        w0, w1, w2, w3 = inv_mass[a], inv_mass[b], inv_mass[c], inv_mass[d]
        sum_w = w0 * g0.norm_sqr() + w1 * g1.norm_sqr() + w2 * g2.norm_sqr(
        ) + w3 * g3.norm_sqr()

        if (sum_w == 0.0):
            continue
        lambda_ = -constriaint / (sum_w + alpha_hydrostatic[i])
        
        pos[a] += w0 * lambda_ * g0
        pos[b] += w1 * lambda_ * g1
        pos[c] += w2 * lambda_ * g2
        pos[d] += w3 * lambda_ * g3


@ti.kernel
def update_velocity(h: ti.f32):
    for i in vel:
        vel[i] = (pos[i] - old_pos[i]) / h


@ti.kernel
def collision_response():
    boundary = 2.0
    for i in pos:
        for j in ti.static(range(3)):
            if (pos[i][j] <= -boundary):
                pos[i][j] = -boundary
            if (pos[i][j] >= boundary):
                pos[i][j] = boundary


if __name__ == "__main__":
    init(pos_np, tet_np, tri_np)
    init_phy(density)
    init_alpha(h)
    window = ti.ui.Window('3D NeoHooean FEM XPBD', (1300, 900), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3.5)
    camera.lookat(0, 0, 0)
    camera.fov(100)
    scene.point_light((0.5, 1.5, 1.5), (1.0, 1.0, 1.0))

    while window.running:
        scene.ambient_light((0.8, 0.8, 0.8))
        camera.track_user_inputs(window,
                                 movement_speed=0.01,
                                 hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        if window.is_pressed(ti.ui.ESCAPE):
            window.running = False

        semi_euler(h)
        for i in range(maxIte):
            project_constraints()
        collision_response()
        update_velocity(h)

        scene.mesh(pos, tri, color=(1.0, 0.5, 0.5), show_wireframe=True)

        canvas.scene(scene)
        window.show()
