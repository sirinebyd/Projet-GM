"""
Vérification formelle d'un réseau de Petri pour système de freinage ferroviaire (ERTMS/ETCS)
Étape 5 et 6 du projet – Simulation et vérification automatique
"""

# Importation d'outils Python standards nécessaires pour le code
from typing import Dict, List, Set, Tuple, Optional  # Pour préciser les types de données (aide à la lecture)
from collections import deque  # Pour utiliser une file d'attente (utile pour l'exploration de l'arbre d'états)
import itertools  # Outils mathématiques pour les itérations
import random  # Pour choisir des actions au hasard lors de la simulation automatique


# ==============================================================================
# PARTIE 1 : LE "MOTEUR" DU RÉSEAU DE PETRI (La logique mathématique de base)
# ==============================================================================

# DÉFINITION D'UNE "PLACE" (Les cercles du réseau)
class Place:
    # La fonction __init__ est le constructeur. Elle s'exécute quand on crée une nouvelle place.
    def __init__(self, name: str, initial_tokens: int = 0):
        self.name = name  # Le nom de la place (ex: "Marche_Normale")
        self.initial_tokens = initial_tokens  # Le nombre de jetons qu'on y met au tout début
        self.tokens = initial_tokens  # Le nombre de jetons actuels (qui va évoluer pendant la simulation)

    # Fonction pour afficher joliment la place dans la console si on l'imprime
    def __repr__(self):
        return f"Place({self.name}, tokens={self.tokens})"


# DÉFINITION D'UNE "TRANSITION" (Les rectangles / portes tournantes du réseau)
class Transition:
    def __init__(self, name: str):
        self.name = name
        # Dictionnaires (collections de paires Clé/Valeur) pour stocker les liaisons
        self.input_arcs: Dict[Place, int] = {}  # Stocke les places AVANT la transition et le nombre de jetons requis
        self.output_arcs: Dict[Place, int] = {}  # Stocke les places APRÈS la transition et le nombre de jetons à créer

    # Fonction pour vérifier si la transition a le droit de s'activer ("tirer")
    def is_enabled(self) -> bool:
        # On parcourt toutes les places d'entrée reliées à cette transition
        for place, weight in self.input_arcs.items():
            # S'il manque des jetons dans au moins une place, la transition est bloquée (False)
            if place.tokens < weight:
                return False
        # Si la boucle se termine sans problème, c'est que toutes les conditions sont réunies (True)
        return True

    # Fonction qui exécute l'action (déplace les jetons)
    def fire(self):
        # 1. On "avale" (soustrait) les jetons des places d'entrée
        for place, weight in self.input_arcs.items():
            place.tokens -= weight
        # 2. On "recrache" (additionne) les jetons dans les places de sortie
        for place, weight in self.output_arcs.items():
            place.tokens += weight


# DÉFINITION DU RÉSEAU COMPLET (Le plateau de jeu qui regroupe Places et Transitions)
class PetriNet:
    def __init__(self):
        self.places: Dict[str, Place] = {}  # Dictionnaire de toutes les places du réseau
        self.transitions: Dict[str, Transition] = {}  # Dictionnaire de toutes les transitions
        self.initial_marking: Dict[str, int] = {}  # Sauvegarde de la position de départ (pour pouvoir recommencer)

    # Fonction pour créer une nouvelle place et l'ajouter au plateau
    def add_place(self, name: str, initial_tokens: int = 0):
        self.places[name] = Place(name, initial_tokens)
        self.initial_marking[name] = initial_tokens  # On mémorise la situation initiale

    # Fonction pour créer une nouvelle transition et l'ajouter au plateau
    def add_transition(self, name: str):
        self.transitions[name] = Transition(name)

    # Fonction pour tracer une flèche (arc) entre un cercle et un rectangle
    def add_arc(self, from_name: str, to_name: str, weight: int = 1, arc_type: str = "input"):
        if arc_type == "input":
            # Flèche de la Place VERS la Transition
            place = self.places[from_name]
            trans = self.transitions[to_name]
            trans.input_arcs[place] = weight  # On enregistre la flèche dans les données de la transition
        elif arc_type == "output":
            # Flèche de la Transition VERS la Place
            trans = self.transitions[from_name]
            place = self.places[to_name]
            trans.output_arcs[place] = weight
        else:
            raise ValueError("arc_type must be 'input' or 'output'")  # Erreur de sécurité si on se trompe de mot

    # Fonction pour tout remettre à zéro avant une nouvelle simulation
    def reset(self):
        for name, place in self.places.items():
            place.tokens = self.initial_marking[name]

    # Fonction qui fouille toutes les transitions et renvoie une liste de celles qui peuvent s'activer
    def get_enabled_transitions(self) -> List[str]:
        return [t.name for t in self.transitions.values() if t.is_enabled()]

    # Fonction pour déclencher une transition spécifique par son nom
    def fire(self, trans_name: str) -> bool:
        trans = self.transitions.get(trans_name)
        if trans and trans.is_enabled():
            trans.fire()
            return True
        return False

    # Fonction utilitaire : compresse l'état du plateau en une liste fixe (Tuple) pour le comparer facilement
    def get_marking_tuple(self) -> Tuple[int, ...]:
        return tuple(self.places[p].tokens for p in sorted(self.places.keys()))

    # Fonction utilitaire : renvoie l'état du plateau sous forme de dictionnaire lisible
    def get_marking_dict(self) -> Dict[str, int]:
        return {name: self.places[name].tokens for name in self.places}

    # MÉTHODE DE L'ÉTAPE 5 : Joue au jeu tout seul pendant X étapes
    def simulate_auto(self, max_steps: int = 100, random_choice: bool = False):
        self.reset()  # Remise à zéro
        trace = [self.get_marking_dict()]  # On crée un historique ("trace") et on y met l'état de départ
        for step in range(max_steps):
            enabled = self.get_enabled_transitions()  # On regarde les actions possibles
            if not enabled:  # S'il n'y a plus d'actions, le système est bloqué, on arrête la simulation
                break
            # On choisit l'action à faire (soit au hasard, soit toujours la première de la liste)
            if random_choice:
                choice = random.choice(enabled)
            else:
                choice = enabled[0]
            self.fire(choice)  # On exécute l'action
            trace.append(self.get_marking_dict())  # On enregistre le nouvel état dans l'historique
        return trace

    # MÉTHODE DE L'ÉTAPE 6 (Model Checking) : Calcule absolument TOUS les futurs possibles
    def build_state_space(self) -> Tuple[Set[Tuple[int, ...]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
        self.reset()
        start = self.get_marking_tuple()
        visited = set()  # Un ensemble pour retenir les états qu'on a déjà explorés
        queue = deque([start])  # Une file d'attente pour explorer le réseau façon "tache d'huile" (Parcours BFS)
        graph = {}  # Le dictionnaire qui va contenir l'arbre de tous les chemins possibles

        while queue:
            state = queue.popleft()  # On prend un état dans la file
            if state in visited:
                continue  # Si on le connaît déjà, on passe au suivant
            visited.add(state)  # On marque cet état comme "connu"
            self.set_marking(state)  # On place physiquement les jetons du réseau dans cet état pour le tester

            succ_states = []  # Liste pour retenir où on peut aller depuis cet état
            for trans in self.transitions.values():
                if trans.is_enabled():
                    # Petite astuce : on sauvegarde les jetons avant de tester la transition
                    old_tokens = {p: p.tokens for p in trans.input_arcs.keys() | trans.output_arcs.keys()}
                    trans.fire()  # On teste l'action
                    new_state = self.get_marking_tuple()  # On mémorise le résultat
                    succ_states.append(new_state)
                    # On annule l'action (on remet les jetons comme avant) pour pouvoir tester les autres transitions
                    for p, tokens in old_tokens.items():
                        p.tokens = tokens
                    # Si ce nouveau futur est inconnu, on l'ajoute à la file d'attente pour l'explorer plus tard
                    if new_state not in visited:
                        queue.append(new_state)
            graph[state] = succ_states  # On relie l'état à tous ses futurs possibles dans le graphe
        return visited, graph

    def set_marking(self, marking_tuple: Tuple[int, ...]):
        """Restaure la disposition des jetons à partir d'une sauvegarde."""
        sorted_places = sorted(self.places.keys())
        for i, name in enumerate(sorted_places):
            self.places[name].tokens = marking_tuple[i]

    # TEST PROPRIÉTÉ P5 : Vérifie l'absence d'interblocage (Deadlock)
    def check_deadlock(self, state_space: Set[Tuple[int, ...]],
                       graph: Dict[Tuple[int, ...], List[Tuple[int, ...]]]) -> bool:
        # On fouille tous les états de l'univers possible
        for state in state_space:
            # Si un état n'a aucun futur (graph[state] est vide), c'est un Deadlock.
            if not graph[state]:
                return False
        return True  # Si on a tout vérifié sans rien trouver, le système est "Deadlock-free"

    # TEST PROPRIÉTÉ P3 : Vérifie les invariants matériels
    def check_invariant(self, coeffs: Dict[str, int], constant: int) -> bool:
        state_space, _ = self.build_state_space()
        sorted_places = sorted(self.places.keys())
        # Pour tous les états possibles, on fait la somme des jetons pour vérifier qu'elle égale la constante (ex: 1)
        for state in state_space:
            total = 0
            for i, p in enumerate(sorted_places):
                total += coeffs.get(p, 0) * state[i]
            if total != constant:
                return False  # Dès qu'un invariant est brisé, on renvoie une erreur
        return True

    # TEST VIVACITÉ : Vérifie qu'aucune transition n'est inutile (morte)
    def check_liveness(self, state_space: Set[Tuple[int, ...]], graph: Dict[Tuple[int, ...], List[Tuple[int, ...]]]) -> \
    Dict[str, bool]:
        # On part du principe que toutes les transitions sont bloquées (False)
        trans_live = {tname: False for tname in self.transitions}
        # On parcourt tous les états
        for state in state_space:
            self.set_marking(state)
            enabled = self.get_enabled_transitions()
            # Si une transition est déclenchable dans cet état, on la valide (True)
            for t in enabled:
                trans_live[t] = True
        return trans_live


    # TEST PROPRIÉTÉ BORNITUDE : Vérifie la limite maximale de jetons dans le réseau
    def check_bornitude(self, state_space: Set[Tuple[int, ...]]) -> int:
        # On initialise le compteur de jetons maximum à 0
        max_tokens = 0
        # On parcourt tous les états possibles dans l'univers du réseau
        for state in state_space:
            # On cherche le nombre de jetons dans la place la plus chargée de cet état
            current_max = max(state)
            # Si cet état contient plus de jetons que notre record précédent, on met à jour le record
            if current_max > max_tokens:
                max_tokens = current_max
        # On renvoie la valeur K, qui définit la "bornitude" du réseau (Réseau K-borné)
        return max_tokens


# ==============================================================================
# PARTIE 2 : NOTRE RÉSEAU DE PETRI (L'Étape 3 du rapport instanciée)
# ==============================================================================

def build_railway_braking_net() -> PetriNet:
    net = PetriNet()  # On fabrique le plateau vierge

    # --- 1. Création des Places (Les États du système ERTMS) ---
    net.add_place("Marche_Normale", 1)
    net.add_place("Alerte_Vitesse", 0)
    net.add_place("Freinage_Service", 0)
    net.add_place("Freinage_Urgence", 0)
    net.add_place("Train_Arrete", 0)
    net.add_place("Radio_OK", 1)
    net.add_place("Radio_Perdue", 0)
    net.add_place("Odometrie_OK", 1)
    net.add_place("Derive_Odo", 0)

    # --- 2. Création des Transitions (Les Actions) ---
    transitions = [
        "T_Survitesse", "T_Conducteur_Ralentit", "T_Ignorer_Alerte", "T_Echec_Service",
        "T_Arret_Complet", "T_Reprise_Marche",
        "T_Perte_Liaison", "T_Reparation_Radio",
        "T_Erreur_Capteur", "T_Recalage_Balise",
        "T_Urgence_Radio_Marche", "T_Urgence_Radio_Alerte", "T_Urgence_Radio_Service",
        "T_Urgence_Odo_Marche", "T_Urgence_Odo_Alerte", "T_Urgence_Odo_Service"
    ]
    for t in transitions:
        net.add_transition(t)

    # --- 3. Arcs Entrants (Conditions : Place -> Transition) ---
    arcs_input = [
        # Boucles standards
        ("Marche_Normale", "T_Survitesse"),
        ("Alerte_Vitesse", "T_Conducteur_Ralentit"),
        ("Alerte_Vitesse", "T_Ignorer_Alerte"),
        ("Freinage_Service", "T_Echec_Service"),
        ("Freinage_Urgence", "T_Arret_Complet"),
        ("Train_Arrete", "T_Reprise_Marche"),
        ("Radio_OK", "T_Perte_Liaison"),
        ("Radio_Perdue", "T_Reparation_Radio"),
        ("Odometrie_OK", "T_Erreur_Capteur"),
        ("Derive_Odo", "T_Recalage_Balise"),
        # Logique de sécurité "Fail-Safe" pour la Radio (Nécessite le train en mouvement ET la panne)
        ("Marche_Normale", "T_Urgence_Radio_Marche"), ("Radio_Perdue", "T_Urgence_Radio_Marche"),
        ("Alerte_Vitesse", "T_Urgence_Radio_Alerte"), ("Radio_Perdue", "T_Urgence_Radio_Alerte"),
        ("Freinage_Service", "T_Urgence_Radio_Service"), ("Radio_Perdue", "T_Urgence_Radio_Service"),
        # Logique de sécurité "Fail-Safe" pour l'Odométrie (Nécessite le train en mouvement ET la dérive)
        ("Marche_Normale", "T_Urgence_Odo_Marche"), ("Derive_Odo", "T_Urgence_Odo_Marche"),
        ("Alerte_Vitesse", "T_Urgence_Odo_Alerte"), ("Derive_Odo", "T_Urgence_Odo_Alerte"),
        ("Freinage_Service", "T_Urgence_Odo_Service"), ("Derive_Odo", "T_Urgence_Odo_Service")
    ]
    for place, trans in arcs_input:
        net.add_arc(place, trans, weight=1, arc_type="input")

    # --- 4. Arcs Sortants (Résultats : Transition -> Place) ---
    arcs_output = [
        ("T_Survitesse", "Alerte_Vitesse"),
        ("T_Conducteur_Ralentit", "Marche_Normale"),
        ("T_Ignorer_Alerte", "Freinage_Service"),
        ("T_Echec_Service", "Freinage_Urgence"),
        ("T_Arret_Complet", "Train_Arrete"),
        ("T_Reprise_Marche", "Marche_Normale"),
        ("T_Perte_Liaison", "Radio_Perdue"),
        ("T_Reparation_Radio", "Radio_OK"),
        ("T_Erreur_Capteur", "Derive_Odo"),
        ("T_Recalage_Balise", "Odometrie_OK"),
        # Les mécanismes d'urgence poussent le jeton vers "Freinage_Urgence" ET restituent le jeton de panne (mémoire de l'erreur)
        ("T_Urgence_Radio_Marche", "Freinage_Urgence"), ("T_Urgence_Radio_Marche", "Radio_Perdue"),
        ("T_Urgence_Radio_Alerte", "Freinage_Urgence"), ("T_Urgence_Radio_Alerte", "Radio_Perdue"),
        ("T_Urgence_Radio_Service", "Freinage_Urgence"), ("T_Urgence_Radio_Service", "Radio_Perdue"),

        ("T_Urgence_Odo_Marche", "Freinage_Urgence"), ("T_Urgence_Odo_Marche", "Derive_Odo"),
        ("T_Urgence_Odo_Alerte", "Freinage_Urgence"), ("T_Urgence_Odo_Alerte", "Derive_Odo"),
        ("T_Urgence_Odo_Service", "Freinage_Urgence"), ("T_Urgence_Odo_Service", "Derive_Odo")
    ]
    for trans, place in arcs_output:
        net.add_arc(trans, place, weight=1, arc_type="output")

    return net


# ==============================================================================
# PARTIE 3 : L'EXÉCUTION DU PROGRAMME (Le point d'entrée principal)
# ==============================================================================

# Cette condition signifie : "Si on exécute ce fichier directement (Bouton Run), on lance les tests."
if __name__ == "__main__":
    net = build_railway_braking_net()  # Instanciation du réseau de l'EVC

    # EXÉCUTION DE L'ÉTAPE 5 : Simulation automatique
    print("=== ÉTAPE 5 : Simulation automatique du réseau (10 étapes) ===")
    trace = net.simulate_auto(max_steps=10, random_choice=True)
    # enumerate permet d'obtenir à la fois le numéro d'étape 'i' et les données 'marking'
    for i, marking in enumerate(trace):
        print(f"Étape {i}: {marking}")

    # EXÉCUTION DE L'ÉTAPE 6 : Vérification Mathématique Exhaustive
    print("\n=== ÉTAPE 6 : Vérification formelle (Model Checking) ===")
    print("Exploration de l'arbre des états en cours...")
    states, graph = net.build_state_space()
    print(f"-> Succès ! Il y a {len(states)} scénarios uniques possibles et aucune explosion combinatoire.")

    # Validation automatisée des propriétés exprimées en CTL/LTL dans le rapport
    no_deadlock = net.check_deadlock(states, graph)
    print(f"\nPropriété P5 (Absence totale de Deadlock) : {'VALIDÉE' if no_deadlock else 'ÉCHEC'}")

    invariant_radio = net.check_invariant({"Radio_OK": 1, "Radio_Perdue": 1}, 1)
    invariant_odo = net.check_invariant({"Odometrie_OK": 1, "Derive_Odo": 1}, 1)
    print(f"Propriété P3 (Invariant Matériel Radio) : {'VALIDÉE' if invariant_radio else 'ÉCHEC'}")
    print(f"Propriété P3 (Invariant Matériel Odométrie) : {'VALIDÉE' if invariant_odo else 'ÉCHEC'}")

    liveness = net.check_liveness(states, graph)
    all_live = all(liveness.values())
    print(f"\nPropriété Vivacité Globale (Aucune transition n'est inutile) : {'VALIDÉE' if all_live else 'ÉCHEC'}")

    k = net.check_bornitude(states)
    print(f"Propriété Bornitude : VALIDÉE (Réseau {k}-borné)")
