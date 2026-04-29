import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _normal_pdf(xs: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * xs ** 2) / np.sqrt(2.0 * np.pi)


def plot_standard_normal_curve(
    output_path: str,
    dpi: int = 900,
    x_min: float = -4.0,
    x_max: float = 4.0,
    num_points: int = 1200,
    transparent_background: bool = False,
) -> None:
    xs = np.linspace(x_min, x_max, num_points, dtype=np.float64)
    ys = _normal_pdf(xs)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=dpi)
    if transparent_background:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    ax.plot(xs, ys, color="#1F77B4", linewidth=8.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, float(np.max(ys)) * 1.02)
    ax.set_box_aspect(1.0)
    ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(
        output_path,
        format="png",
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor="none" if transparent_background else "white",
        transparent=transparent_background,
    )
    plt.close(fig)
    print(f"[Figure] Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a standard normal curve as a square PNG.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/analysis/standard_normal_curve.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=900, help="Output dpi.")
    parser.add_argument("--x_min", type=float, default=-4.0, help="Minimum x value.")
    parser.add_argument("--x_max", type=float, default=4.0, help="Maximum x value.")
    parser.add_argument("--num_points", type=int, default=1200, help="Number of sampled points.")
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save the PNG with a transparent background.",
    )
    args = parser.parse_args()

    if args.x_min >= args.x_max:
        raise ValueError(f"x_min must be smaller than x_max, got {args.x_min} and {args.x_max}.")
    if args.num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {args.num_points}.")

    plot_standard_normal_curve(
        output_path=args.output_path,
        dpi=args.dpi,
        x_min=args.x_min,
        x_max=args.x_max,
        num_points=args.num_points,
        transparent_background=args.transparent,
    )


if __name__ == "__main__":
    main()
