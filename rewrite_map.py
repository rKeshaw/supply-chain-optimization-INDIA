import re
import sys
from pathlib import Path

file_path = Path("frontend/index.html")
content = file_path.read_text(encoding="utf-8")

# 1. Add CDNs to head
cdn_tags = """  <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/deck.gl@8.9.32/dist.min.js"></script>"""
content = content.replace("</title>", "</title>\n" + cdn_tags)

# 2. CSS for map
css_old = """    #network-svg {
      width: 100%;
      height: 100%;
    }"""
css_new = """    #map {
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      border-radius: 12px;
    }
    /* Prevent DeckGL mapbox logo from overlapping controls */
    .maplibregl-control-container { display: none; }"""
content = content.replace(css_old, css_new)

# 3. HTML for map
html_old = """        <svg id="network-svg">
          <!-- Defined nodes and links dynamically via JS -->
        </svg>"""
html_new = """        <div id="map"></div>"""
content = content.replace(html_old, html_new)

# 4. JS: Initialize DeckGL and replace renderGraph
js_old = """    // ----------------------------------------------------
    // SVG Network Rendering
    // ----------------------------------------------------
    function renderGraph(graph) {"""

js_split = content.split(js_old)
if len(js_split) != 2:
    print("Could not find start of renderGraph.")
    sys.exit(1)

js_top = js_split[0]
js_bottom = js_split[1]

action_handlers_start = "    // ----------------------------------------------------\n    // Action Handlers\n    // ----------------------------------------------------"
bottom_split = js_bottom.split(action_handlers_start)
if len(bottom_split) != 2:
    print("Could not find start of Action Handlers.")
    sys.exit(1)

js_bottom_rest = action_handlers_start + bottom_split[1]

new_deck_js = """    // ----------------------------------------------------
    // Deck.GL Geospatial Rendering
    // ----------------------------------------------------
    let deckgl = null;

    function renderGraph(graph) {
      APP_STATE.graph = graph;
      
      const nodesData = graph.nodes.filter(n => n.id !== "super_source" && n.id !== "super_sink" && n.lat && n.lon);
      
      const edgesData = graph.edges.filter(e => {
        if (e.from_id === "super_source" || e.to_id === "super_sink") return false;
        const src = graph.nodes.find(n => n.id === e.from_id);
        const tgt = graph.nodes.find(n => n.id === e.to_id);
        if (!src || !tgt || !src.lat || !tgt.lat) return false;
        
        const effectiveOpenness = e.effective_openness ?? e.openness ?? 1.0;
        
        // Hide edges with completely closed corridors to fix "edges still passing through" issue
        if (effectiveOpenness <= 0.01) return false;
        
        e.src = src;
        e.tgt = tgt;
        return true;
      });

      // Scatterplot Layer for Nodes
      const nodeLayer = new deck.ScatterplotLayer({
        id: 'nodes-layer',
        data: nodesData,
        pickable: true,
        opacity: 0.9,
        stroked: true,
        filled: true,
        radiusScale: 1000,
        radiusMinPixels: 4,
        radiusMaxPixels: 15,
        lineWidthMinPixels: 2,
        getPosition: d => [d.lon, d.lat],
        getRadius: d => {
          if (d.type === "chokepoint") return 80;
          if (d.type.startsWith("refinery")) return 60;
          return 50;
        },
        getFillColor: d => {
          if (d.type === "spr") return [16, 185, 129];
          if (d.type.startsWith("refinery")) return [139, 92, 246]; // Purple
          if (d.openness < 0.2) return [239, 68, 68];
          if (d.openness < 0.8) return [245, 158, 11];
          return [59, 130, 246];
        },
        getLineColor: d => [255, 255, 255, 100],
        onClick: (info) => {
          if (info.object) clickNode(info.object.id);
        }
      });

      // Arc Layer for Edges (Flow)
      const arcLayer = new deck.ArcLayer({
        id: 'arcs-layer',
        data: edgesData,
        pickable: true,
        getWidth: d => {
          const flow = d.flow_bbl_day || 0;
          const flowRatio = flow / Math.max(d.base_capacity_bbl_day || 1, 1);
          return Math.max(2, flowRatio * 10);
        },
        getSourcePosition: d => [d.src.lon, d.src.lat],
        getTargetPosition: d => [d.tgt.lon, d.tgt.lat],
        getSourceColor: d => {
          if ((d.effective_openness ?? 1) < 0.8) return [239, 68, 68, 200];
          if (d.mode === 'pipeline') return [16, 185, 129, 200];
          return [59, 130, 246, 200];
        },
        getTargetColor: d => {
          if ((d.effective_openness ?? 1) < 0.8) return [239, 68, 68, 200];
          if (d.mode === 'pipeline') return [16, 185, 129, 200];
          return [59, 130, 246, 200];
        },
        onClick: (info) => {
          if (info.object) clickEdge(info.object.id);
        }
      });

      if (!deckgl) {
        deckgl = new deck.DeckGL({
          container: 'map',
          mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
          initialViewState: {
            longitude: 65.0,
            latitude: 20.0,
            zoom: 3.2,
            pitch: 45,
            bearing: 0
          },
          controller: true,
          layers: [arcLayer, nodeLayer],
          getTooltip: ({object}) => {
             if (!object) return null;
             if (object.name) {
               return `${object.name}\\nRisk: ${(object.risk_score*100).toFixed(0)}%\\nAvailable: ${(object.openness*100).toFixed(0)}%`;
             }
             if (object.id) {
               return `${object.id}\\nFlow: ${(object.flow_bbl_day/1000).toFixed(0)}k bpd\\nAvailable: ${( (object.effective_openness ?? 1)*100).toFixed(0)}%`;
             }
             return null;
          }
        });
      } else {
        deckgl.setProps({
          layers: [arcLayer, nodeLayer]
        });
      }
    }
"""

content = js_top + new_deck_js + "\n" + js_bottom_rest
file_path.write_text(content, encoding="utf-8")
print("Successfully modified index.html!")
