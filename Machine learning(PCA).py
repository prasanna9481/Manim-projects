from manim import *
import numpy as np

class CombinedPCAAndFancyScene(Scene):
    def construct(self):
        # Call the different functions in the order you want them to be executed
        self.intro()
        self.pca_equations()
        self.pca_2d_demo()
        self.fancy_shape_scene()
        self.projection_error_scene()
        self.references()
        self.thank_you()

    def intro(self):
        title = Text("Principal Component Analysis (PCA)", font_size=48)
        subtitle = Text("Maximum Variance & Minimum Error", font_size=32)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=DOWN))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        self.play(Write(subtitle))
        self.wait(2)

        # Fade out the title before displaying the equations
        self.play(FadeOut(title), FadeOut(subtitle))

    def pca_equations(self):
        # PCA Equations (Mean, Covariance, Eigen Decomp)
        
        # (a) Mean / Centering
        mean_eq = MathTex(
            r"\boldsymbol{\mu} = \frac{1}{N}\sum_{n=1}^{N} \mathbf{x}_n"
        ).scale(0.8)
        mean_eq.to_edge(UP, buff=1)

        centered_eq = MathTex(
            r"\mathbf{X}_{\mathrm{centered}} = \mathbf{X} - \boldsymbol{\mu}"
        ).scale(0.8)
        centered_eq.next_to(mean_eq, DOWN, buff=0.6)

        self.play(Write(mean_eq))
        self.wait(1)
        self.play(Write(centered_eq))
        self.wait(1)

        # (b) Maximum Variance Formulation
        max_var_title = Text("Maximum-Variance Formulation", font_size=34, color=YELLOW)
        max_var_title.next_to(centered_eq, DOWN, buff=0.8)
        self.play(Write(max_var_title))
        self.wait()

        cov_eq = MathTex(
            r"\mathbf{S} = \frac{1}{N}\sum_{n=1}^{N} (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T"
        ).scale(0.68)
        cov_eq.next_to(max_var_title, DOWN, buff=0.6)  
        self.play(Write(cov_eq))
        self.wait(2)

        variance_eq = MathTex(
            r"\mathrm{Var}(\mathbf{X}\text{-proj}) = \mathbf{u}^T \mathbf{S} \,\mathbf{u}"
        ).scale(0.8)
        variance_eq.next_to(cov_eq, DOWN, buff=0.4)

        max_eq = MathTex(
            r"\max_{\mathbf{u}} \;\mathbf{u}^T \mathbf{S}\,\mathbf{u} \quad"
            r"\text{subject to} \quad \|\mathbf{u}\|=1"
        ).scale(0.7)
        max_eq.next_to(variance_eq, DOWN, buff=0.4)

        self.play(Write(variance_eq))
        self.wait(1)
        self.play(Write(max_eq))
        self.wait(1)

        eigen_eq = MathTex(
            r"\mathbf{S}\,\mathbf{u} = \lambda \mathbf{u} \quad\Longrightarrow\quad"
            r"\mathbf{u} \text{ is eigenvector, } \lambda \text{ is eigenvalue.}"
        ).scale(0.7)
        eigen_eq.next_to(max_eq, DOWN, buff=0.5)

        self.play(Write(eigen_eq))
        self.wait(2)

        min_err_title = Text("Minimum-Error Formulation", font_size=34, color=GREEN)
        min_err_title.next_to(eigen_eq, DOWN, buff=1.0)
        self.play(Write(min_err_title))
        self.wait(1)

        error_eq = MathTex(
            r"J = \frac{1}{N}\sum_{n=1}^{N} \|\mathbf{x}_n - \mathbf{\tilde{x}}_n\|^2"
        ).scale(0.8)
        error_eq.next_to(min_err_title, DOWN, buff=0.4)
        self.play(Write(error_eq))
        self.wait(1)

        eq_conclusion = MathTex(
            r"\min J \;\Longleftrightarrow\; \text{top eigenvectors of } \mathbf{S}."
        ).scale(0.8)
        eq_conclusion.next_to(error_eq, DOWN, buff=0.5)
        self.play(Write(eq_conclusion))
        self.wait(2)

        recap_text = Text(
            "PCA => Project onto top-k eigenvectors, retaining maximum variance",
            font_size=30
        )
        recap_text.next_to(eq_conclusion, DOWN, buff=0.8)
        self.play(FadeIn(recap_text))
        self.wait(1)

        # Fade out equations
        self.play(
            *[FadeOut(mob) for mob in [
                mean_eq, centered_eq, cov_eq, max_var_title,
                variance_eq, max_eq, eigen_eq, min_err_title,
                error_eq, eq_conclusion, recap_text
            ]]
        )

    def pca_2d_demo(self):
        demo_title = Text("2D PCA Demonstration", font_size=36)
        self.play(Write(demo_title))
        self.wait(1)
        self.play(demo_title.animate.to_edge(UP))
        self.play(FadeOut(demo_title))

        # Axes
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), FadeIn(axes_labels))

        # Define the data points
        np.random.seed(42)
        data_pc1 = 3 * np.random.randn(50)
        data_pc2 = 0.5 * np.random.randn(50)
        data = np.column_stack([data_pc1, data_pc2])

        points = VGroup(*[
            Dot(axes.coords_to_point(x, y), radius=0.06, color=BLUE)
            for x, y in data
        ])
        self.play(Create(points))

        # Compute the mean and mark it
        mean = np.mean(data, axis=0)
        mean_dot = Dot(axes.coords_to_point(*mean), color=RED, radius=0.1)
        mean_label = Tex("Mean").next_to(mean_dot, UP)
        self.play(FadeIn(mean_dot), Write(mean_label))
        self.wait(1)

        # Animate shift to the origin
        shift_vector = axes.coords_to_point(0, 0) - axes.coords_to_point(*mean)
        self.play(
            points.animate.shift(shift_vector),
            mean_dot.animate.shift(shift_vector),
            run_time=2
        )

        # After shifting, update the label
        new_label = Tex("Mean at (0,0)").next_to(mean_dot, UP)
        self.play(Transform(mean_label, new_label))
        self.wait(1)

        # Create PC1 and PC2 vectors
        pc1_vector = np.array([5, 0])
        pc2_vector = np.array([0, 3])
        scale_factor = 3

        pc1_line = Line(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(*(pc1_vector * scale_factor)),
            color=RED
        ).set_stroke(width=5)
        
        pc2_line = Line(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(*(pc2_vector * scale_factor)),
            color=GREEN
        ).set_stroke(width=5)

        # Add labels for PC1 and PC2
        pc1_label = Tex("PC1").next_to(pc1_line, RIGHT)
        pc2_label = Tex("PC2").next_to(pc2_line, UP)

        self.play(Create(pc1_line), Create(pc2_line))
        self.play(Write(pc1_label), Write(pc2_label))

        # Arrows for variance
        variance_arrow_pc1 = Arrow(
            start=axes.coords_to_point(0, 0),
            end=axes.coords_to_point(*(pc1_vector * scale_factor * 0.7)),
            color=RED
        )
        
        variance_arrow_pc2 = Arrow(
            start=axes.coords_to_point(0, 0),
            end=axes.coords_to_point(*(pc2_vector * scale_factor * 0.7)),
            color=GREEN
        )

        self.play(Create(variance_arrow_pc1), Create(variance_arrow_pc2))

        variance_text_pc1 = Tex("Variance along PC1").next_to(variance_arrow_pc1, DOWN, buff=0.2).shift(LEFT * 0.5)
        variance_text_pc2 = Tex("Variance along PC2").next_to(variance_arrow_pc2, UP)

        self.play(Write(variance_text_pc1), Write(variance_text_pc2))

        transform_text = Tex(
            "Transforming 2D to 1D while preserving variance", font_size=24
        ).next_to(axes, DOWN)
        self.play(Write(transform_text))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in [
            axes, axes_labels, points, mean_dot, mean_label,
            pc1_line, pc2_line, pc1_label, pc2_label,
            variance_arrow_pc1, variance_arrow_pc2, variance_text_pc1, variance_text_pc2, transform_text
        ]])

    def fancy_shape_scene(self):
        shape = Circle(radius=2, color=BLUE).set_fill(RED, opacity=0.5)
        shape.shift(UP)

        self.play(Create(shape))

        text = Tex("using minimum error").move_to(shape.get_center())
        self.play(Write(text))

        self.play(FadeOut(shape), FadeOut(text))

        box = Square(side_length=4)
        box.set_fill(WHITE, opacity=0.5)
        box.shift(DOWN)
        self.play(Create(box))

        term1 = Tex(r"$\tilde{x}_n = \sum_{d=1}^{M} z_n d u_d$")
        term2 = Tex(r"$ + \sum_{d=M+1}^{D} b_d u_d$")

        term1.move_to(box.get_center() + UP*0.5)
        term2.move_to(box.get_center() + DOWN*0.5)

        self.play(Write(term1))
        self.play(Write(term2))

        self.play(term1.animate.shift(LEFT*2), term2.animate.shift(RIGHT*2))
        
        self.wait(1)
        self.play(FadeOut(term1), FadeOut(term2), FadeOut(box))

    def projection_error_scene(self):
        heading = Text("Projection error after transformation").scale(0.8)
        heading.shift(UP * 3)

        self.play(Write(heading))

        formula = Tex(r"$J = \frac{1}{N} \sum_{n=1}^{N} \|x_n - \tilde{x}_n\|^2$")
        formula.scale(1)  

        box = Rectangle(width=6, height=3)
        box.set_fill(WHITE, opacity=0.5)

        box.move_to(heading.get_center() + DOWN * 2)  
        formula.move_to(box.get_center())

        self.play(Create(box))
        self.play(Write(formula))

        self.play(
            box.animate.shift(LEFT * 6),  
            formula.animate.shift(LEFT * 6)
        )

        arrow = Arrow(start=box.get_edge_center(RIGHT), end=ORIGIN, buff=0.5)
        self.play(Create(arrow))

        next_formula = Tex(r"$\min_{u_d} u_d^T S u_d + \lambda_2(1 - u_2^T u_2)$")
        next_formula.scale(1)

        new_box = Rectangle(width=6, height=3)
        new_box.set_fill(WHITE, opacity=0.5)
        new_box.move_to(ORIGIN)  

        next_formula.move_to(new_box.get_center())

        self.play(Create(new_box))
        self.play(Write(next_formula))

        self.play(FadeOut(heading), FadeOut(formula), FadeOut(box), FadeOut(arrow), FadeOut(new_box), FadeOut(next_formula))

    def references(self):
        ref_title = Text("References", font_size=40).to_edge(UP)
        self.play(Write(ref_title))
        self.wait(1)

        gfg_text = Text(
            "1) GeeksforGeeks: 'Principal Component Analysis (PCA)'",
            font_size=28
        ).next_to(ref_title, DOWN, buff=1)
        self.play(Write(gfg_text))
        self.wait(1)

        pdf_text = Text(
            "2) UTN ML LU7-3 Principal Component Analysis.pdf",
            font_size=28
        ).next_to(gfg_text, DOWN, buff=0.5)
        self.play(Write(pdf_text))
        self.wait(1)

        bishop_text = Text(
            "3) Bishop, 'Pattern Recognition and Machine Learning', Chapter 12.1",
            font_size=28
        ).next_to(pdf_text, DOWN, buff=0.5)
        self.play(Write(bishop_text))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()

    def thank_you(self):
        star = Star(
            n=5,  
            color=YELLOW,
            stroke_width=2,
            fill_opacity=0.8,
        )
        star.scale(3)
        star.move_to(ORIGIN)

        thank_you_text = Text("Thank You", font_size=48)
        thank_you_text.move_to(star.get_center())

        self.play(Create(star))
        self.play(Write(thank_you_text))

        self.wait(2)
        self.play(FadeOut(star), FadeOut(thank_you_text))
