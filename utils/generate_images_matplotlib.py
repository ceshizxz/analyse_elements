import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import string
from shapely.geometry import LineString, Polygon, Point, MultiPoint

class ShapeImageGenerator:
    def __init__(self, output_dir, num_images, shape_range, file_prefix, width=800, height=800):
        self.output_dir = output_dir
        self.num_images = num_images
        self.shape_range = shape_range
        self.file_prefix = file_prefix
        self.width = width
        self.height = height
        self.figsize = (8, 8)
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_random_shape(self, ax, lines):
        shape_type = random.choice(["line", "rectangle", "trapezoid", "triangle", "circle", "semicircle"])
        color = "black"
        linewidth = 1.0
        center_x, center_y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
        size = random.uniform(0.4, 0.7)

        if shape_type == "line":
            x1, y1 = random.uniform(0, 1), random.uniform(0, 1)
            x2, y2 = random.uniform(0, 1), random.uniform(0, 1)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
            lines.append(LineString([(x1, y1), (x2, y2)]))
        
        elif shape_type == "circle":
            radius = size * 0.3
            circle = patches.Circle((center_x, center_y), radius, edgecolor=color, facecolor='none', linewidth=linewidth)
            ax.add_patch(circle)
            lines.append(Point(center_x, center_y).buffer(radius).boundary)
        
        elif shape_type == "semicircle":
            radius = size * 0.3
            angle = random.choice([0, 90, 180, 270])
            semicircle = patches.Wedge((center_x, center_y), radius, angle, angle + 180, edgecolor=color, facecolor='none', linewidth=linewidth)
            ax.add_patch(semicircle)
            lines.append(Point(center_x, center_y).buffer(radius, resolution=10).boundary)
        
        elif shape_type in ["rectangle", "trapezoid", "triangle"]:
            w, h = size * 0.5, size * 0.5
            if shape_type == "rectangle":
                points = [(center_x - w/2, center_y - h/2),
                          (center_x + w/2, center_y - h/2),
                          (center_x + w/2, center_y + h/2),
                          (center_x - w/2, center_y + h/2)]
            elif shape_type == "trapezoid":
                top_w = random.uniform(0.3, 0.5) * w
                points = [(center_x - w/2, center_y + h/2),
                          (center_x + w/2, center_y + h/2),
                          (center_x + top_w/2, center_y - h/2),
                          (center_x - top_w/2, center_y - h/2)]
            elif shape_type == "triangle":
                points = [(center_x, center_y - h/2),
                          (center_x - w/2, center_y + h/2),
                          (center_x + w/2, center_y + h/2)]

            ax.add_patch(plt.Polygon(points, edgecolor=color, facecolor='none', linewidth=linewidth))
            poly_boundary = LineString(points + [points[0]])
            lines.append(poly_boundary)

    def find_intersections(self, lines):
        intersections = set()
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i].intersects(lines[j]):
                    inter = lines[i].intersection(lines[j])
                    if isinstance(inter, Point):
                        intersections.add((inter.x, inter.y))
                    elif isinstance(inter, MultiPoint):
                        for p in inter.geoms:
                            intersections.add((p.x, p.y))
        return list(intersections)

    def annotate_intersections(self, ax, intersections, lines):
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        shift_options = [(-0.03, 0), (0.03, 0), (0, -0.03), (0, 0.03)] 
        
        for i, (x, y) in enumerate(intersections[:26]):
            for _ in range(20): 
                shift_x, shift_y = random.choice(shift_options)
                new_x = min(max(x + shift_x, 0.05), 0.95)
                new_y = min(max(y + shift_y, 0.05), 0.95)
                
                if not any(line.intersects(Point(new_x, new_y)) for line in lines):
                    ax.text(new_x, new_y, letters[i % len(letters)], fontsize=6, ha='center', va='center', color='black', fontweight='bold')
                    break 

    def generate_shape_image(self, img_index):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        fig.patch.set_facecolor('white')

        lines = []
        num_shapes = random.randint(*self.shape_range)
        for _ in range(num_shapes):
            self.draw_random_shape(ax, lines)

        intersections = self.find_intersections(lines)
        if len(intersections) > 20:
            plt.close(fig)
            return
        if len(intersections) < 1:
            plt.close(fig)
            return  

        self.annotate_intersections(ax, intersections, lines)
        folder_name = str(len(intersections))
        save_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        file_name = f"{self.file_prefix}{img_index}.png"
        fig.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def generate_dataset(self):
        for i in range(1, self.num_images + 1):
            self.generate_shape_image(i)


output_dir = r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\images\generated_images"
os.makedirs(output_dir, exist_ok=True)

generator1 = ShapeImageGenerator(output_dir, 5000, (1, 3), "a")
generator1.generate_dataset()

generator2 = ShapeImageGenerator(output_dir, 6000, (4, 6), "b")
generator2.generate_dataset()

generator3 = ShapeImageGenerator(output_dir, 3000, (7, 8), "c")
generator3.generate_dataset()

print("Done!")