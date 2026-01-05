"""
Module subriemannian_geometry.py - Géométrie sous-riemannienne sur l'espace de contact
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Callable
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class SubRiemannianGeometry:
    """
    Géométrie sous-riemannienne définie sur la structure de contact.
    La métrique est définie seulement sur les plans de contact.
    """
    
    def __init__(self, spatial_shape: Tuple[int, int], device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du champ visuel
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.device = device
        
        # Métrique sous-riemannienne standard sur le plan de contact
        # En coordonnées (x, y, p), le plan de contact est engendré par:
        # X = ∂/∂x + p ∂/∂y
        # P = ∂/∂p
        # Métrique: g(X,X) = 1, g(P,P) = 1, g(X,P) = 0
        
        self.metric_tensor = self._create_standard_metric()
        
    def _create_standard_metric(self) -> Callable:
        """
        Crée la métrique sous-riemannienne standard.
        
        Returns:
            Fonction g(point, vector1, vector2) → produit scalaire
        """
        def metric(point: torch.Tensor, 
                   vector1: torch.Tensor, 
                   vector2: torch.Tensor) -> float:
            """
            Calcule le produit scalaire sous-riemannien.
            
            Args:
                point: (x, y, p) point de l'espace
                vector1, vector2: Vecteurs tangents
                
            Returns:
                Produit scalaire
            """
            # Pour la métrique standard:
            # Seulement la composante dans le plan de contact compte
            
            # Projette les vecteurs sur le plan de contact
            v1_proj = self.project_to_contact_plane(point, vector1)
            v2_proj = self.project_to_contact_plane(point, vector2)
            
            # Produit scalaire euclidien des projections
            return torch.dot(v1_proj, v2_proj).item()
        
        return metric
    
    def project_to_contact_plane(self, point: torch.Tensor, 
                                vector: torch.Tensor) -> torch.Tensor:
        """
        Projette un vecteur tangent sur le plan de contact.
        
        Args:
            point: (x, y, p)
            vector: (vx, vy, vp)
            
        Returns:
            Vecteur projeté
        """
        x, y, p = point
        vx, vy, vp = vector
        
        # Base du plan de contact
        X = torch.tensor([1.0, p, 0.0], device=self.device)  # ∂/∂x + p ∂/∂y
        P = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # ∂/∂p
        
        # Coordonnées dans cette base
        # On résout: vector = a*X + b*P
        # Donc: vx = a, vy = a*p, vp = b
        a = vx
        b = vp
        
        # Vérifie la cohérence
        if abs(vy - a * p) > 1e-6:
            # Le vecteur n'est pas dans le plan, on projette
            # On garde seulement la partie dans le plan
            pass
        
        # Retourne la projection
        return a * X + b * P
    
    def subriemannian_length(self, curve: torch.Tensor) -> float:
        """
        Calcule la longueur sous-riemannienne d'une courbe.
        
        Args:
            curve: Tensor (n_points, 3) points (x, y, p)
            
        Returns:
            Longueur de la courbe
        """
        if curve.shape[0] < 2:
            return 0.0
        
        length = 0.0
        
        for i in range(curve.shape[0] - 1):
            p1 = curve[i]
            p2 = curve[i + 1]
            
            # Vecteur tangent (approximation)
            tangent = p2 - p1
            
            # Vitesse sous-riemannienne
            speed = self.subriemannian_norm(p1, tangent)
            
            # Longueur d'arc euclidienne
            ds = torch.norm(p2[:2] - p1[:2]).item()
            
            length += speed * ds
        
        return length
    
    def subriemannian_norm(self, point: torch.Tensor, 
                          vector: torch.Tensor) -> float:
        """
        Calcule la norme sous-riemannienne d'un vecteur.
        
        Args:
            point: (x, y, p)
            vector: (vx, vy, vp)
            
        Returns:
            Norme sous-riemannienne
        """
        # Projette sur le plan de contact
        proj = self.project_to_contact_plane(point, vector)
        
        # Norme euclidienne de la projection
        return torch.norm(proj).item()
    
    def energy_functional(self, curve: torch.Tensor) -> float:
        """
        Calcule l'énergie sous-riemannienne d'une courbe.
        Utilisé pour les problèmes variationnels.
        
        Args:
            curve: Tensor (n_points, 3)
            
        Returns:
            Énergie de la courbe
        """
        if curve.shape[0] < 2:
            return 0.0
        
        energy = 0.0
        
        for i in range(curve.shape[0] - 1):
            p1 = curve[i]
            p2 = curve[i + 1]
            
            tangent = p2 - p1
            speed_sq = self.subriemannian_norm(p1, tangent) ** 2
            
            # Longueur d'arc
            ds = torch.norm(p2[:2] - p1[:2]).item()
            
            energy += speed_sq * ds
        
        return energy
    
    def hamiltonian(self, point: torch.Tensor, 
                   momentum: torch.Tensor) -> float:
        """
        Hamiltonien sous-riemannien.
        
        Args:
            point: (x, y, p) coordonnées
            momentum: (px, py, pp) moments
            
        Returns:
            Valeur du hamiltonien
        """
        x, y, p = point
        px, py, pp = momentum
        
        # Pour la métrique standard:
        # H = 1/2 * (h1^2 + h2^2)
        # où h1 = px + p*py, h2 = pp
        h1 = px + p * py
        h2 = pp
        
        return 0.5 * (h1**2 + h2**2).item()
    
    def hamiltonian_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Équations de Hamilton pour les géodésiques.
        
        Args:
            t: Temps (paramètre)
            state: [x, y, p, px, py, pp]
            
        Returns:
            Dérivées [dx/dt, dy/dt, dp/dt, dpx/dt, dpy/dt, dpp/dt]
        """
        x, y, p, px, py, pp = state
        
        # Hamiltonien: H = 1/2 * ((px + p*py)^2 + pp^2)
        h1 = px + p * py
        
        # Équations de Hamilton:
        dx_dt = h1
        dy_dt = p * h1
        dp_dt = pp
        
        # Équations pour les moments:
        dpx_dt = 0
        dpy_dt = 0
        dpp_dt = -h1 * py
        
        return np.array([dx_dt, dy_dt, dp_dt, dpx_dt, dpy_dt, dpp_dt])
    
    def integrate_geodesic(self, start_state: np.ndarray,
                          duration: float = 1.0,
                          n_steps: int = 100) -> np.ndarray:
        """
        Intègre les équations de Hamilton pour obtenir une géodésique.
        
        Args:
            start_state: [x0, y0, p0, px0, py0, pp0]
            duration: Durée de l'intégration
            n_steps: Nombre de pas
            
        Returns:
            Trajectoire (n_steps, 6)
        """
        t_eval = np.linspace(0, duration, n_steps)
        
        solution = solve_ivp(
            fun=self.hamiltonian_equations,
            t_span=(0, duration),
            y0=start_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        return solution.y.T
    
    def shooting_method(self, start_point: np.ndarray,
                       end_point: np.ndarray,
                       initial_momentum: np.ndarray = None,
                       max_iter: int = 100) -> np.ndarray:
        """
        Méthode de shooting pour trouver une géodésique entre deux points.
        
        Args:
            start_point: [x0, y0, p0]
            end_point: [x1, y1, p1]
            initial_momentum: [px0, py0, pp0] guess initial
            max_iter: Nombre max d'itérations
            
        Returns:
            Géodésique trouvée
        """
        if initial_momentum is None:
            # Guess initial: momentum aligné avec la direction
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            dp = end_point[2] - start_point[2]
            
            initial_momentum = np.array([dx, dy, dp])
        
        # Fonction coût: distance entre point final atteint et cible
        def cost_function(momentum):
            state = np.concatenate([start_point, momentum])
            trajectory = self.integrate_geodesic(state, duration=1.0)
            
            # Point final atteint
            final_point = trajectory[-1, :3]
            
            # Distance à la cible
            error = np.linalg.norm(final_point - end_point)
            return error
        
        # Optimisation
        result = minimize(
            cost_function,
            initial_momentum,
            method='BFGS',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success:
            optimal_momentum = result.x
            state = np.concatenate([start_point, optimal_momentum])
            geodesic = self.integrate_geodesic(state)
            return geodesic
        else:
            raise RuntimeError(f"Shooting method failed: {result.message}")
    
    def compute_carnot_caratheodory_distance(self, 
                                           point1: torch.Tensor,
                                           point2: torch.Tensor,
                                           n_samples: int = 20) -> float:
        """
        Calcule la distance de Carnot-Carathéodory (approximative).
        C'est la longueur de la géodésique la plus courte.
        
        Args:
            point1, point2: Points (x, y, p)
            n_samples: Nombre d'échantillons pour l'approximation
            
        Returns:
            Distance approximative
        """
        try:
            # Convertit en numpy
            p1_np = point1.cpu().numpy()
            p2_np = point2.cpu().numpy()
            
            # Utilise la méthode de shooting
            geodesic = self.shooting_method(p1_np, p2_np, max_iter=50)
            
            # Calcule la longueur
            length = 0.0
            for i in range(len(geodesic) - 1):
                p1 = torch.tensor(geodesic[i, :3])
                p2 = torch.tensor(geodesic[i + 1, :3])
                tangent = p2 - p1
                speed = self.subriemannian_norm(p1, tangent)
                ds = torch.norm(p2[:2] - p1[:2]).item()
                length += speed * ds
            
            return length
            
        except Exception as e:
            # Fallback: distance euclidienne dans l'espace de contact
            print(f"Shooting method failed, using Euclidean fallback: {e}")
            return torch.norm(point1 - point2).item()
    
    def create_subriemannian_ball(self, center: torch.Tensor,
                                 radius: float,
                                 n_points: int = 1000) -> torch.Tensor:
        """
        Crée une approximation d'une boule sous-riemannienne.
        
        Args:
            center: Centre (x, y, p)
            radius: Rayon
            n_points: Nombre de points à générer
            
        Returns:
            Points à la surface de la boule
        """
        points = []
        
        for _ in range(n_points):
            # Génère une direction aléatoire dans le plan de contact
            a = np.random.randn()  # Coefficient pour X
            b = np.random.randn()  # Coefficient pour P
            
            # Vecteur dans le plan de contact
            x, y, p = center
            X = torch.tensor([1.0, p, 0.0], device=self.device)
            P = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            
            direction = a * X + b * P
            direction = direction / torch.norm(direction)
            
            # Avance dans cette direction
            point = center + radius * direction
            points.append(point)
        
        return torch.stack(points)
