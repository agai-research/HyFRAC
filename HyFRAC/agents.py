"""
agents.py -- the seven agents realised on a pluggable backbone.

"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

from goal_tree import GoalTree, GoalNode, Ontology


@dataclass
class ParsedRequest:
    """Output schema of the Decomposition agent's LlmParse step."""
    objectives: List[str]
    preferences: Dict[str, float]   # normalised over (cost, duration, reliability, complexity)
    budget: Dict[str, float]
    level: int
    modality: str = "any"


class Backbone:
    """
    Single extension point for a real language model. mode='rule_based' (the
    default here) never leaves this process; mode='local_hf' would route
    through a locally-served open-weight model if one is configured in
    Config/backbone.json. Swapping backbones does not touch any algorithm.
    """

    def __init__(self, config: dict):
        self.mode = config.get("mode", "rule_based")
        self.model_name = config.get("model_name", "none")

    def call(self, prompt: str, schema_hint: str = "") -> str:
        if self.mode == "rule_based":
            raise RuntimeError("Backbone.call() should not be reached in rule_based mode; "
                                "use the dedicated agent methods instead.")
        raise NotImplementedError(f"backbone mode '{self.mode}' is not wired up in this prototype")


# --------------------------------------------------------------------------
# Decomposition agent: LlmParse, LlmRepair (Algorithm 1)
# --------------------------------------------------------------------------
CAPABILITY_KEYWORDS = {
    # a keyword table standing in for the language model's free-text
    # understanding; a real backbone would replace only this lookup.
    # covers all 25 education capabilities of Data/capability_catalog.py.
    "broadcasting": ["stream", "broadcast", "display", "screen"],
    "quizzing": ["quiz", "test", "assess"],
    "whiteboarding": ["whiteboard", "collaborate", "board"],
    "ambient sensing": ["sense", "monitor", "occupancy"],
    "ambient regulation": ["climate", "light", "temperature", "regulate"],
    "acoustics": ["audio", "sound", "noise"],
    "polling": ["poll", "vote"],
    "attendance tracking": ["attendance", "roll call", "check-in"],
    "note synchronisation": ["notes", "note-taking", "synchronise notes"],
    "lecture recording": ["record", "recording", "capture lecture"],
    "screen sharing": ["share screen", "screen share", "present"],
    "flashcard generation": ["flashcard", "flash card", "memorise"],
    "peer grouping": ["group", "pair up", "teams"],
    "plagiarism checking": ["plagiarism", "originality", "similarity check"],
    "assignment collection": ["assignment", "submission", "homework"],
    "gradebook synchronisation": ["gradebook", "grades", "grading"],
    "discussion moderation": ["discussion", "forum", "moderate"],
    "live captioning": ["caption", "subtitle", "live text"],
    "language translation": ["translate", "translation", "language"],
    "accessibility adaptation": ["accessibility", "accessible", "adapt"],
    "concept mapping": ["concept map", "mind map", "mapping"],
    "study reminder": ["remind", "reminder", "study schedule"],
    "revision scheduling": ["revision", "review schedule", "recap"],
    "exam proctoring": ["proctor", "exam supervision", "invigilate"],
    "certificate issuance": ["certificate", "certification", "credential"],
}


class DecompositionAgent:
    def __init__(self, ontology: Ontology, backbone: Backbone):
        self.ontology = ontology
        self.backbone = backbone

    def parse(self, request: str) -> ParsedRequest:
        """LlmParse: extract objectives, level, modality, budget from free text."""
        text = request.lower()
        objectives = [cap for cap, kws in CAPABILITY_KEYWORDS.items() if any(k in text for k in kws)]
        if not objectives:
            objectives = [list(CAPABILITY_KEYWORDS.keys())[0]]  # default objective if nothing matched

        level = 1
        m = re.search(r"level\s*(\d)", text)
        if m:
            level = int(m.group(1))
        elif "beginner" in text or "first-year" in text:
            level = 1
        elif "advanced" in text:
            level = 3

        budget = {"cost": 200.0, "duration": 120.0, "complexity": level + 1}
        preferences = {"cost": 0.2, "duration": 0.2, "reliability": 0.4, "complexity": 0.2}
        return ParsedRequest(objectives=objectives, preferences=preferences, budget=budget, level=level)

    def build_goal_tree(self, parsed: ParsedRequest) -> GoalTree:
        """Lines 4-13 of Algorithm 1: AND node per objective with prereqs, OR node for alternatives."""
        tree = GoalTree()
        for objective in parsed.objectives:
            node = self._expand(objective, parsed, visited=set(), depth=0)
            tree.root.children.append(node)
        return tree

    def _expand(self, capability: str, parsed: ParsedRequest, visited: set, depth: int,
                max_depth: int = 3) -> GoalNode:
        """
        visited guards against a prerequisite cycle; max_depth is a second,
        independent safeguard against a deep-but-acyclic chain pulling in
        far more capabilities than a request actually needs (a request
        should require its immediate prerequisites, not a long transitive
        closure). Both stop expansion by treating the capability as a plain
        leaf rather than recursing further.
        """
        if capability in visited or depth >= max_depth:
            return GoalNode(kind="leaf", capability=capability, target_complexity=parsed.level,
                             modality=parsed.modality)
        visited = visited | {capability}

        prereqs = self.ontology.prerequisites(capability)
        alternatives = self.ontology.alternatives(capability)
        if prereqs:
            node = GoalNode(kind="and")
            for p in prereqs:
                node.children.append(self._expand(p, parsed, visited, depth + 1))
            node.children.append(GoalNode(kind="leaf", capability=capability,
                                           target_complexity=parsed.level, modality=parsed.modality))
            return node
        if len(alternatives) > 1:
            node = GoalNode(kind="or")
            for alt in alternatives:
                node.children.append(GoalNode(kind="leaf", capability=alt,
                                               target_complexity=parsed.level, modality=parsed.modality))
            return node
        return GoalNode(kind="leaf", capability=capability,
                         target_complexity=parsed.level, modality=parsed.modality)

    def repair(self, tree: GoalTree, missing: List[str]) -> GoalTree:
        """LlmRepair: attach the missing prerequisite capabilities as extra AND children."""
        for capability in missing:
            tree.root.children.append(GoalNode(kind="leaf", capability=capability))
        return tree


# --------------------------------------------------------------------------
# Mediator agent: LlmMediate (Algorithm 2)
# --------------------------------------------------------------------------
class MediatorAgent:
    def mediate(self, violations: List[tuple], repository, goal_tree: GoalTree) -> list:
        """Propose bridging fragments from the repository that supply a missing post-condition."""
        suggestions = []
        for combination, reason in violations:
            for frag in combination:
                for pred in frag.pre:
                    suggestions.extend(repository.bridging_candidates(pred))
        return list({f.id: f for f in suggestions}.values())


# --------------------------------------------------------------------------
# Planner / Critic / Ranker agents (Algorithm 3)
# --------------------------------------------------------------------------
class PlannerAgent:
    def plan(self, ranked_fragments: list, assignment: list, catalogue: Dict, template=None):
        """
        Builds a draft DAG covering every leaf of the satisfying assignment.
        Fragments are deduplicated, then greedily added by (how many still-
        uncovered leaves they cover, their retrieval/filtering score) so a
        required leaf is not starved out by an earlier, higher-scoring
        fragment that happens to overlap its services with no other fragment
        left able to cover that leaf.

        A fragment derived from a res-share, co-exec, or cap-cover hyperedge
        can legitimately bundle services of capabilities unrelated to any
        current leaf (that is what lets it be checked for resource
        contention at retrieval time); only the subset of a chosen
        fragment's services whose capability is actually required is
        admitted into the delivered application, so an unrelated bundled
        service is never pulled in just because it happened to share a
        hyperedge with a relevant one.
        """
        from optimisation import Application

        required = {leaf.capability for leaf in assignment}
        unique_fragments = list({f.id: f for f in ranked_fragments}.values())

        chosen: List = []
        seen_services: set = set()
        covered: set = set()

        while covered != required:
            best_frag, best_gain = None, -1
            for frag in unique_fragments:
                if frag.id in {c.id for c in chosen}:
                    continue
                relevant_services = {s for s in frag.services
                                     if catalogue.get(s) and catalogue[s].capability in required}
                if relevant_services & seen_services:
                    continue
                gain = len((frag.capabilities & required) - covered)
                if gain > best_gain:
                    best_frag, best_gain = frag, gain
            if best_frag is None or best_gain <= 0:
                break  # no remaining fragment can extend coverage without conflict
            chosen.append(best_frag)
            relevant = {s for s in best_frag.services
                        if catalogue.get(s) and catalogue[s].capability in required}
            seen_services |= relevant
            covered |= (best_frag.capabilities & required)

        return Application.from_fragments(chosen, catalogue, required)


class CriticAgent:
    def critique(self, application, goal_tree: GoalTree, level: int, band_width: int) -> List[str]:
        """C2 (interface), C3 (complexity band), C6 (acyclic + coverage) -- C1/C4 already hold."""
        issues = []
        if not application.is_acyclic():
            issues.append("C6: application is not acyclic")
        required = {leaf.capability for leaf in goal_tree.leaves}
        covered = application.covered_capabilities()
        if not required.issubset(covered):
            issues.append(f"C6: missing capabilities {required - covered}")
        for s in application.services.values():
            if not (level <= s.complexity <= level + band_width):
                issues.append(f"C3: service {s.id} outside complexity band")
        return issues


class RankerAgent:
    def resolve(self, application, issues: List[str], ranked_fragments: list,
                catalogue: Dict, required: set):
        """
        Addresses a C3 (complexity band) issue by replacing the named
        violating service's fragment with a same-capability, in-band
        alternative from ranked_fragments; addresses any other issue by
        adding the next best unused, service-disjoint fragment. A pure
        "add more on top" strategy can never fix a violation caused by a
        service already in the draft, so the two issue types need
        different repairs.
        """
        if not issues:
            return application

        band_issue = next((i for i in issues if i.startswith("C3:")), None)
        if band_issue is not None:
            replaced = self._replace_violating_service(application, band_issue, ranked_fragments,
                                                         catalogue, required)
            if replaced is not None:
                return replaced

        for frag in ranked_fragments:
            if frag.id in application.fragment_ids:
                continue
            relevant = {s for s in frag.services if catalogue.get(s) and catalogue[s].capability in required}
            if relevant & set(application.services.keys()):
                continue  # would duplicate a service already in the draft
            return application.with_fragment_added(frag, catalogue, required)
        return application  # no substitution found -> unchanged, signals a deadlock

    def _replace_violating_service(self, application, band_issue: str, ranked_fragments: list,
                                    catalogue: Dict, required: set):
        """Parses 'C3: service <id> outside complexity band' and swaps that service's
        fragment for a same-capability, in-band alternative, if one exists."""
        parts = band_issue.split()
        violating_id = parts[2] if len(parts) > 2 else None
        if violating_id is None or violating_id not in catalogue:
            return None
        capability = catalogue[violating_id].capability

        for frag in ranked_fragments:
            if frag.id in application.fragment_ids:
                continue
            if capability not in frag.capabilities:
                continue
            replacement_services = {s for s in frag.services if catalogue.get(s)
                                     and catalogue[s].capability == capability}
            if not replacement_services or violating_id in replacement_services:
                continue
            # drop the violating service and any edge touching it; the fragment_ids
            # bookkeeping keeps the original id since no service-to-fragment map is
            # tracked here, a minor metadata imprecision that does not affect which
            # services or edges the application actually contains
            new_services = {s: v for s, v in application.services.items() if s != violating_id}
            new_edges = [(a, b) for a, b in application.edges if a != violating_id and b != violating_id]
            from optimisation import Application
            trimmed = Application(services=new_services, edges=new_edges,
                                   fragment_ids=application.fragment_ids)
            return trimmed.with_fragment_added(frag, catalogue, required)
        return None


# --------------------------------------------------------------------------
# Explanation agent (Algorithm 4)
# --------------------------------------------------------------------------
class ExplanationAgent:
    def verbalise(self, causal_graph, application, register: str) -> str:
        """LlmVerbalise: template-based narration citing every retained causal event."""
        lines = [f"The delivered application uses {len(application.services)} service(s): "
                 f"{', '.join(sorted(application.services))}."]
        for node in causal_graph.nodes:
            lines.append(f"- {node.kind}: {node.summary}")
        return "\n".join(lines)

    def clarify(self, source: str, context: str) -> str:
        """LlmClarify: turns the dominant source of uncertainty into a targeted question."""
        return f"Could you clarify {source}? This affects: {context}."
