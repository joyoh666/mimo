import matplotlib.pyplot as plt
import numpy as np

import sionna.rt
from sionna.rt import Camera, Transmitter, Receiver, PlanarArray, PathSolver, load_scene


def main():
    # 1) 간단한 예제 씬 로드
    scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                       merge_shapes=True)

    # 2) 송신기와 수신기를 모두 MIMO로 구성
    scene.tx_array = PlanarArray(
        num_rows=2,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V"
    )

    tx = Transmitter(name="tx",
                     position=[22.7, 5.6, 0.75],
                     orientation=[np.pi, 0, 0])
    rx = Receiver(name="rx",
                  position=[45.0, 90.0, 1.5],
                  orientation=[0.0, 0.0, 0.0])
    scene.add(tx)
    scene.add(rx)

    # 3) PathSolver로 MIMO 경로 계산
    solver = PathSolver()
    print("--- Computing MIMO paths ---")
    paths = solver(
        scene=scene,
        max_depth=3,
        max_num_paths_per_src=200000,
        samples_per_src=20000,
        synthetic_array=False,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        seed=0
    )

    # 4) preview 렌더링
    cam = Camera(position=[20.0, 40.0, 20.0], look_at=[30.0, 50.0, 5.0])
    print("--- Rendering preview with scene and MIMO paths ---")

    # Interactive preview (if supported in your environment)
    scene.preview(paths=paths,
                  show_devices=True,
                  show_orientations=True,
                  resolution=(900, 600))

    # Static preview image
    fig = scene.render(camera=cam,
                       paths=paths,
                       resolution=(900, 600),
                       num_samples=512,
                       show_devices=True,
                       show_orientations=True)

    preview_path = "sionna_mimo_preview.png"
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    print(f"Preview saved to {preview_path}")

    # 5) 송신기/수신기 위치 정보 및 경로 요약
    print(f"TX name: {tx.name}, position: {tx.position.numpy().tolist()}")
    print(f"RX name: {rx.name}, position: {rx.position.numpy().tolist()}")
    print("paths.valid shape:", paths.valid.shape)
    print("paths.interactions shape:", paths.interactions.shape)
    print("paths.objects shape:", paths.objects.shape)

    plt.show()


if __name__ == "__main__":
    main()
