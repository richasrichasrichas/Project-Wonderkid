"""
main.py

KivyMD front-end for the scouting pipeline. Works on both desktop and
mobile from the same codebase (Kivy's whole point).

Two screens:
  - PlayerListScreen: search + scrollable list of players, pulled from
    the CSV produced by the scraper/overall/potential pipeline.
  - PlayerDetailScreen: breakdown of a single player — Base Overall,
    Current-Form Overall, Estimated Potential, and raw attributes.

If no CSV is found at DATA_CSV_PATH, falls back to a small bundled
sample dataset so the app runs standalone for development/demo.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional

import pandas as pd
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.list import MDListItem
from kivymd.uix.boxlayout import MDBoxLayout


DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_players.csv")

# Columns the app expects in the CSV. This matches the output shape of
# the player_overall.py / player_potential.py / sports_scraper pipeline.
REQUIRED_COLUMNS = [
    "player_name", "position", "club", "age",
    "base_overall", "current_form_overall", "estimated_potential",
]
ATTRIBUTE_COLUMNS = [
    "finishing", "passing_vision", "dribbling",
    "tackling", "physical", "pace", "positioning",
]


KV = """
#:import dp kivy.metrics.dp


<PlayerListItem>:
    theme_bg_color: "Custom"
    md_bg_color: self.theme_cls.surfaceContainerLowColor
    radius: [12, 12, 12, 12]
    padding: dp(4)

    MDListItemLeadingIcon:
        icon: "soccer"

    MDListItemHeadlineText:
        text: root.player_name

    MDListItemSupportingText:
        text: root.subtitle_text

    MDListItemTrailingSupportingText:
        text: root.overall_text
        theme_text_color: "Custom"
        text_color: root.overall_color
        bold: True


<PlayerListScreen>:
    name: "player_list"

    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            type: "small"

            MDTopAppBarLeadingButtonContainer:
                MDActionTopAppBarButton:
                    icon: "soccer-field"

            MDTopAppBarTitle:
                text: "Scouting - Base de Jogadores"

        MDBoxLayout:
            orientation: "vertical"
            padding: dp(12), dp(8), dp(12), dp(8)
            spacing: dp(8)
            size_hint_y: None
            height: dp(64)

            MDTextField:
                id: search_field
                mode: "filled"
                on_text: root.on_search_text(self.text)

                MDTextFieldLeadingIcon:
                    icon: "magnify"

                MDTextFieldHintText:
                    text: "Buscar jogador por nome, posicao ou clube"

        ScrollView:
            MDList:
                id: player_list_container
                padding: dp(8)
                spacing: dp(6)


<PlayerDetailScreen>:
    name: "player_detail"

    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            type: "small"

            MDTopAppBarLeadingButtonContainer:
                MDActionTopAppBarButton:
                    icon: "arrow-left"
                    on_release: root.go_back()

            MDTopAppBarTitle:
                text: root.player_name

        ScrollView:
            MDBoxLayout:
                id: detail_container
                orientation: "vertical"
                padding: dp(16)
                spacing: dp(16)
                adaptive_height: True
"""


class PlayerListItem(MDListItem):
    """One row in the player list: name, position/club subtitle, overall badge."""
    player_name = StringProperty("")
    subtitle_text = StringProperty("")
    overall_text = StringProperty("")
    overall_color = ListProperty([0, 0, 0, 1])
    player_id = NumericProperty(0)

    def on_release(self):
        app = MDApp.get_running_app()
        app.open_player_detail(self.player_id)


def overall_color_for(value: float) -> List[float]:
    """Green for high overall, amber for mid, red for low — quick visual scan."""
    if value >= 75:
        return [0.20, 0.60, 0.30, 1]   # green
    if value >= 60:
        return [0.80, 0.60, 0.10, 1]   # amber
    return [0.70, 0.20, 0.20, 1]       # red


class PlayerListScreen(MDScreen):
    def populate(self, players: pd.DataFrame) -> None:
        container = self.ids.player_list_container
        container.clear_widgets()
        for idx, row in players.iterrows():
            item = PlayerListItem(
                player_name=str(row["player_name"]),
                subtitle_text=f"{row['position']} - {row['club']} - {int(row['age'])} anos",
                overall_text=f"{row['base_overall']:.0f}",
                overall_color=overall_color_for(row["base_overall"]),
                player_id=int(idx),
            )
            container.add_widget(item)

    def on_search_text(self, text: str) -> None:
        MDApp.get_running_app().filter_players(text)


class AttributeBar(MDBoxLayout):
    """Simple labeled progress row for one attribute (0-100)."""
    def __init__(self, label: str, value: float, **kwargs):
        super().__init__(orientation="vertical", spacing=2, adaptive_height=True, **kwargs)
        from kivymd.uix.label import MDLabel
        from kivymd.uix.progressindicator import MDLinearProgressIndicator

        row = MDBoxLayout(orientation="horizontal", adaptive_height=True)
        row.add_widget(MDLabel(text=label, adaptive_height=True, size_hint_x=0.6))
        row.add_widget(MDLabel(text=f"{value:.0f}", adaptive_height=True,
                                halign="right", size_hint_x=0.4))
        self.add_widget(row)

        bar = MDLinearProgressIndicator(value=value, max=100)
        self.add_widget(bar)


class OverallSummaryCard(MDBoxLayout):
    """Card-style block showing the three headline numbers: base, form, potential."""
    def __init__(self, base_ov: float, form_ov: float, potential: float, **kwargs):
        super().__init__(orientation="vertical", spacing=8, adaptive_height=True,
                          padding=12, **kwargs)
        from kivymd.uix.card import MDCard
        from kivymd.uix.label import MDLabel

        card = MDCard(orientation="horizontal", padding=16, spacing=16,
                       size_hint_y=None, height="110dp", radius=[16, 16, 16, 16])
        for label, value, color in [
            ("Overall Geral", base_ov, overall_color_for(base_ov)),
            ("Overall de Fase", form_ov, overall_color_for(form_ov)),
            ("Potencial", potential, overall_color_for(potential)),
        ]:
            col = MDBoxLayout(orientation="vertical")
            value_label = MDLabel(text=f"{value:.0f}", halign="center",
                                   font_style="Headline", bold=True)
            value_label.theme_text_color = "Custom"
            value_label.text_color = color
            col.add_widget(value_label)
            col.add_widget(MDLabel(text=label, halign="center", font_style="Label"))
            card.add_widget(col)
        self.add_widget(card)


class PlayerDetailScreen(MDScreen):
    player_name = StringProperty("Jogador")

    def show_player(self, row: pd.Series) -> None:
        self.player_name = str(row["player_name"])
        container = self.ids.detail_container
        container.clear_widgets()

        container.add_widget(OverallSummaryCard(
            base_ov=float(row["base_overall"]),
            form_ov=float(row["current_form_overall"]),
            potential=float(row["estimated_potential"]),
        ))

        from kivymd.uix.label import MDLabel
        container.add_widget(MDLabel(text="Atributos", font_style="Title", adaptive_height=True))
        for attr in ATTRIBUTE_COLUMNS:
            if attr in row and pd.notna(row[attr]):
                container.add_widget(AttributeBar(label=attr.replace("_", " ").title(),
                                                    value=float(row[attr])))

    def go_back(self) -> None:
        self.manager.current = "player_list"


def load_players(csv_path: str = DATA_CSV_PATH) -> pd.DataFrame:
    """
    Loads the player dataset. Falls back to a small bundled sample if the
    pipeline hasn't produced a real CSV yet, so the UI is runnable on its
    own during development.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = _sample_dataset()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV at {csv_path} is missing required columns: {missing}")
    return df.reset_index(drop=True)


def _sample_dataset() -> pd.DataFrame:
    """Small fictitious dataset used only when no real pipeline CSV exists yet."""
    return pd.DataFrame([
        {"player_name": "Jogador Exemplo A", "position": "FWD", "club": "Clube Ficticio SC",
         "age": 21, "base_overall": 78, "current_form_overall": 80, "estimated_potential": 85,
         "finishing": 82, "passing_vision": 65, "dribbling": 78, "tackling": 30,
         "physical": 70, "pace": 85, "positioning": 75},
        {"player_name": "Jogador Exemplo B", "position": "MID", "club": "Atletico Ficticio",
         "age": 26, "base_overall": 74, "current_form_overall": 71, "estimated_potential": 76,
         "finishing": 55, "passing_vision": 84, "dribbling": 70, "tackling": 60,
         "physical": 65, "pace": 62, "positioning": 78},
        {"player_name": "Jogador Exemplo C", "position": "CB", "club": "Uniao Ficticia FC",
         "age": 30, "base_overall": 81, "current_form_overall": 83, "estimated_potential": 81,
         "finishing": 20, "passing_vision": 60, "dribbling": 30, "tackling": 88,
         "physical": 85, "pace": 55, "positioning": 84},
    ])


class ScoutingApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.theme_style = "Dark"

        self.players: pd.DataFrame = load_players()
        self.filtered_players: pd.DataFrame = self.players

        self.screen_manager = ScreenManager()
        self.list_screen = PlayerListScreen()
        self.detail_screen = PlayerDetailScreen()
        self.screen_manager.add_widget(self.list_screen)
        self.screen_manager.add_widget(self.detail_screen)
        return self.screen_manager

    def on_start(self):
        self.list_screen.populate(self.filtered_players)

    def filter_players(self, query: str) -> None:
        query = query.strip().lower()
        if not query:
            self.filtered_players = self.players
        else:
            mask = (
                self.players["player_name"].str.lower().str.contains(query)
                | self.players["position"].str.lower().str.contains(query)
                | self.players["club"].str.lower().str.contains(query)
            )
            self.filtered_players = self.players[mask]
        self.list_screen.populate(self.filtered_players)

    def open_player_detail(self, player_id: int) -> None:
        row = self.filtered_players.loc[player_id]
        self.detail_screen.show_player(row)
        self.screen_manager.current = "player_detail"


if __name__ == "__main__":
    Builder.load_string(KV)
    ScoutingApp().run()