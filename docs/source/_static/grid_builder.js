/**
 * GridAges Interactive Drag-and-Drop Microgrid & DER Builder
 * Generates runnable GridAges Python environments dynamically.
 */

(function () {
  const PRESETS = {
    ieee13: {
      name: "IEEE 13-Bus Microgrid",
      envClass: "IEEE13Env",
      networkFunc: "IEEE13Bus",
      buses: [
        { id: "650", name: "Bus 650 (Source / Substation)", kv: 115.0, isSlack: true },
        { id: "632", name: "Bus 632", kv: 4.16 },
        { id: "671", name: "Bus 671 (Main Feed)", kv: 4.16 },
        { id: "680", name: "Bus 680", kv: 4.16 },
        { id: "675", name: "Bus 675 (Distributed)", kv: 4.16 },
        { id: "692", name: "Bus 692", kv: 4.16 },
        { id: "652", name: "Bus 652", kv: 4.16 },
        { id: "611", name: "Bus 611", kv: 4.16 },
        { id: "684", name: "Bus 684", kv: 4.16 },
      ],
      defaultDevices: [
        { id: "ess_1", type: "battery", name: "Battery_671", bus: "671", capacity_mwh: 1.0, max_p_mw: 0.5, efficiency: 0.95 },
        { id: "pv_1", type: "solar", name: "SolarPV_675", bus: "675", max_p_mw: 0.8 },
        { id: "dg_1", type: "generator", name: "DieselDG_680", bus: "680", min_p_mw: 0.1, max_p_mw: 0.6, startup_cost: 15.0 },
      ]
    },
    cigre_mv: {
      name: "CIGRE Medium Voltage (MV)",
      envClass: "CIGREMVEnv",
      networkFunc: "create_cigre_network_mv",
      buses: [
        { id: "Bus 1", name: "Bus 1 (HV/MV Interconnect)", kv: 20.0, isSlack: true },
        { id: "Bus 3", name: "Bus 3", kv: 20.0 },
        { id: "Bus 5", name: "Bus 5", kv: 20.0 },
        { id: "Bus 7", name: "Bus 7", kv: 20.0 },
        { id: "Bus 9", name: "Bus 9", kv: 20.0 },
        { id: "Bus 11", name: "Bus 11", kv: 20.0 },
        { id: "Bus 14", name: "Bus 14", kv: 20.0 },
      ],
      defaultDevices: [
        { id: "ess_cigre", type: "battery", name: "CIGRE_ESS", bus: "Bus 5", capacity_mwh: 2.0, max_p_mw: 1.0, efficiency: 0.92 },
        { id: "wind_1", type: "wind", name: "WindFarm_Bus7", bus: "Bus 7", max_p_mw: 1.5 },
      ]
    },
    custom3: {
      name: "Modular 3-Bus Islanded Microgrid",
      envClass: "GridBaseEnv",
      networkFunc: "create_custom_network",
      buses: [
        { id: "Bus 1", name: "Bus 1 (Slack/PCC)", kv: 13.8, isSlack: true },
        { id: "Bus 2", name: "Bus 2 (Feeder A)", kv: 13.8 },
        { id: "Bus 3", name: "Bus 3 (Feeder B)", kv: 13.8 },
      ],
      defaultDevices: [
        { id: "pv_custom", type: "solar", name: "RooftopPV", bus: "Bus 2", max_p_mw: 0.4 },
        { id: "ess_custom", type: "battery", name: "CommunityBattery", bus: "Bus 3", capacity_mwh: 0.8, max_p_mw: 0.4, efficiency: 0.96 },
      ]
    }
  };

  const DEVICE_TYPES = {
    battery: {
      name: "Battery Storage (ESS)",
      icon: "🔋",
      iconClass: "icon-battery",
      desc: "gridages.devices.ESS",
      defaultProps: { capacity_mwh: 1.0, max_p_mw: 0.5, efficiency: 0.95 }
    },
    solar: {
      name: "Solar PV (RES)",
      icon: "☀️",
      iconClass: "icon-solar",
      desc: "gridages.devices.RES (solar)",
      defaultProps: { max_p_mw: 0.6 }
    },
    wind: {
      name: "Wind Turbine (RES)",
      icon: "💨",
      iconClass: "icon-wind",
      desc: "gridages.devices.RES (wind)",
      defaultProps: { max_p_mw: 1.2 }
    },
    generator: {
      name: "Gas / Diesel DG",
      icon: "🏭",
      iconClass: "icon-gen",
      desc: "gridages.devices.DG",
      defaultProps: { min_p_mw: 0.1, max_p_mw: 0.8, startup_cost: 20.0 }
    },
    load: {
      name: "Flexible / Critical Load",
      icon: "⚡",
      iconClass: "icon-load",
      desc: "Active / Reactive Demand",
      defaultProps: { p_mw: 0.4, q_mvar: 0.1 }
    }
  };

  class GridBuilderApp {
    constructor(rootElem) {
      this.root = rootElem;
      this.currentPresetKey = "ieee13";
      this.devices = JSON.parse(JSON.stringify(PRESETS[this.currentPresetKey].defaultDevices));
      this.selectedDevice = null;
      this.counter = 10;
      this.render();
    }

    render() {
      const preset = PRESETS[this.currentPresetKey];
      this.root.innerHTML = `
        <div class="gb-header">
          <div class="gb-title-group">
            <h3>⚡ GridAges Interactive Microgrid & DER Builder</h3>
            <p class="gb-subtitle">Drag DERs from the palette onto grid buses. Live Python environment code is generated automatically.</p>
          </div>
          <div class="gb-controls">
            <label for="gb-preset-select" style="font-size:0.8rem; font-weight:600; color:var(--gb-text-muted);">Topology:</label>
            <select id="gb-preset-select" class="gb-select">
              <option value="ieee13" ${this.currentPresetKey === "ieee13" ? "selected" : ""}>IEEE 13-Bus Radial Feeder</option>
              <option value="cigre_mv" ${this.currentPresetKey === "cigre_mv" ? "selected" : ""}>CIGRE Medium Voltage</option>
              <option value="custom3" ${this.currentPresetKey === "custom3" ? "selected" : ""}>Modular 3-Bus Microgrid</option>
            </select>
            <button class="gb-btn gb-btn-outline" id="gb-reset-btn" title="Reset to Defaults">↺ Reset</button>
          </div>
        </div>

        <div class="gb-workspace">
          <!-- Left: Draggable DER Palette -->
          <div class="gb-palette">
            <div class="gb-palette-title">📦 DER Component Palette</div>
            ${Object.entries(DEVICE_TYPES).map(([typeKey, dev]) => `
              <div class="gb-device-card" draggable="true" data-type="${typeKey}">
                <div class="gb-device-icon ${dev.iconClass}">${dev.icon}</div>
                <div class="gb-device-meta">
                  <span class="gb-device-name">${dev.name}</span>
                  <span class="gb-device-desc">${dev.desc}</span>
                </div>
              </div>
            `).join("")}
          </div>

          <!-- Right: Grid Bus Canvas -->
          <div class="gb-canvas-area">
            <div class="gb-canvas-helper">
              <span>📍 <strong>${preset.name}</strong> (${preset.buses.length} Buses)</span>
              <span>💡 Tip: Drag a device onto any bus card below</span>
            </div>
            <div class="gb-grid-diagram" id="gb-canvas-buses">
              ${preset.buses.map(bus => this.renderBusNode(bus)).join("")}
            </div>
          </div>
        </div>

        <!-- Generated Code Section -->
        <div class="gb-code-section">
          <div class="gb-code-header">
            <span class="gb-code-header-title">🐍 Generated GridAges Python Code</span>
            <div>
              <button class="gb-btn gb-btn-primary" id="gb-copy-btn">📋 Copy Code</button>
            </div>
          </div>
          <pre class="gb-code-block"><code id="gb-code-content">${this.generatePythonCode()}</code></pre>
        </div>

        <!-- Modal Container -->
        <div id="gb-modal-host"></div>
      `;

      this.bindEvents();
    }

    renderBusNode(bus) {
      const busDevices = this.devices.filter(d => d.bus === bus.id);
      return `
        <div class="gb-bus-node" data-bus-id="${bus.id}">
          <div class="gb-bus-header">
            <span class="gb-bus-title">
              ${bus.isSlack ? "⚡" : "🚏"} ${bus.id}
            </span>
            <span class="gb-bus-badge">${bus.kv} kV</span>
          </div>
          <div class="gb-bus-devices">
            ${busDevices.length === 0 ? `
              <div class="gb-empty-hint">Drop DER here</div>
            ` : busDevices.map(dev => {
              const meta = DEVICE_TYPES[dev.type] || { icon: "⚙️", name: dev.type };
              const summary = dev.type === "battery" ? `${dev.capacity_mwh} MWh` :
                              dev.type === "generator" ? `${dev.max_p_mw} MW` :
                              dev.type === "load" ? `${dev.p_mw} MW` : `${dev.max_p_mw} MW`;
              return `
                <div class="gb-placed-item" data-dev-id="${dev.id}" title="Click to edit parameters">
                  <span class="gb-placed-info">
                    <span>${meta.icon}</span>
                    <span>${dev.name} (${summary})</span>
                  </span>
                  <button class="gb-remove-btn" data-remove-id="${dev.id}" title="Remove device">×</button>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    }

    bindEvents() {
      // Preset Selector
      const select = this.root.querySelector("#gb-preset-select");
      if (select) {
        select.addEventListener("change", (e) => {
          this.currentPresetKey = e.target.value;
          this.devices = JSON.parse(JSON.stringify(PRESETS[this.currentPresetKey].defaultDevices));
          this.render();
        });
      }

      // Reset Button
      const resetBtn = this.root.querySelector("#gb-reset-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          this.devices = JSON.parse(JSON.stringify(PRESETS[this.currentPresetKey].defaultDevices));
          this.render();
        });
      }

      // Copy Code Button
      const copyBtn = this.root.querySelector("#gb-copy-btn");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => {
          const code = this.generatePythonCode();
          navigator.clipboard.writeText(code).then(() => {
            copyBtn.textContent = "✓ Copied!";
            setTimeout(() => { copyBtn.textContent = "📋 Copy Code"; }, 2000);
          });
        });
      }

      // Setup Drag from Palette
      const dragCards = this.root.querySelectorAll(".gb-device-card");
      dragCards.forEach(card => {
        card.addEventListener("dragstart", (e) => {
          card.classList.add("dragging");
          e.dataTransfer.setData("text/plain", card.dataset.type);
        });
        card.addEventListener("dragend", () => {
          card.classList.remove("dragging");
        });
      });

      // Setup Drop Targets on Buses
      const busNodes = this.root.querySelectorAll(".gb-bus-node");
      busNodes.forEach(node => {
        node.addEventListener("dragover", (e) => {
          e.preventDefault();
          node.classList.add("drag-over");
        });
        node.addEventListener("dragleave", () => {
          node.classList.remove("drag-over");
        });
        node.addEventListener("drop", (e) => {
          e.preventDefault();
          node.classList.remove("drag-over");
          const devType = e.dataTransfer.getData("text/plain");
          const busId = node.dataset.busId;
          if (devType && busId) {
            this.addDevice(devType, busId);
          }
        });
      });

      // Device click for edit modal & remove button
      const canvasArea = this.root.querySelector("#gb-canvas-buses");
      if (canvasArea) {
        canvasArea.addEventListener("click", (e) => {
          const removeBtn = e.target.closest(".gb-remove-btn");
          if (removeBtn) {
            e.stopPropagation();
            const devId = removeBtn.dataset.removeId;
            this.devices = this.devices.filter(d => d.id !== devId);
            this.render();
            return;
          }

          const placedItem = e.target.closest(".gb-placed-item");
          if (placedItem) {
            const devId = placedItem.dataset.devId;
            const dev = this.devices.find(d => d.id === devId);
            if (dev) this.openEditModal(dev);
          }
        });
      }
    }

    addDevice(type, busId) {
      this.counter++;
      const info = DEVICE_TYPES[type];
      const newDev = {
        id: `${type}_${this.counter}`,
        type: type,
        name: `${type.toUpperCase()}_${busId}`,
        bus: busId,
        ...info.defaultProps
      };
      this.devices.push(newDev);
      this.render();
    }

    openEditModal(dev) {
      const modalHost = this.root.querySelector("#gb-modal-host");
      modalHost.innerHTML = `
        <div class="gb-modal-overlay">
          <div class="gb-modal-content">
            <div class="gb-modal-header">
              <h4 style="margin:0;">⚙️ Edit ${dev.name}</h4>
              <button class="gb-remove-btn" id="gb-modal-close" style="font-size:1.2rem;">×</button>
            </div>
            <form id="gb-edit-form">
              <div class="gb-form-row">
                <label>Device Name</label>
                <input type="text" name="name" value="${dev.name}" required />
              </div>
              <div class="gb-form-row">
                <label>Connected Bus</label>
                <input type="text" name="bus" value="${dev.bus}" required />
              </div>
              ${dev.type === "battery" ? `
                <div class="gb-form-row">
                  <label>Storage Capacity (MWh)</label>
                  <input type="number" step="0.1" name="capacity_mwh" value="${dev.capacity_mwh || 1.0}" />
                </div>
                <div class="gb-form-row">
                  <label>Max Power (MW)</label>
                  <input type="number" step="0.1" name="max_p_mw" value="${dev.max_p_mw || 0.5}" />
                </div>
                <div class="gb-form-row">
                  <label>Round-Trip Efficiency</label>
                  <input type="number" step="0.01" max="1.0" min="0.5" name="efficiency" value="${dev.efficiency || 0.95}" />
                </div>
              ` : ""}
              ${dev.type === "generator" ? `
                <div class="gb-form-row">
                  <label>Min Power (MW)</label>
                  <input type="number" step="0.05" name="min_p_mw" value="${dev.min_p_mw || 0.1}" />
                </div>
                <div class="gb-form-row">
                  <label>Max Power (MW)</label>
                  <input type="number" step="0.05" name="max_p_mw" value="${dev.max_p_mw || 0.8}" />
                </div>
              ` : ""}
              ${dev.type === "solar" || dev.type === "wind" ? `
                <div class="gb-form-row">
                  <label>Rated Peak Capacity (MW)</label>
                  <input type="number" step="0.1" name="max_p_mw" value="${dev.max_p_mw || 0.5}" />
                </div>
              ` : ""}
              ${dev.type === "load" ? `
                <div class="gb-form-row">
                  <label>Active Demand (MW)</label>
                  <input type="number" step="0.05" name="p_mw" value="${dev.p_mw || 0.4}" />
                </div>
                <div class="gb-form-row">
                  <label>Reactive Demand (MVar)</label>
                  <input type="number" step="0.05" name="q_mvar" value="${dev.q_mvar || 0.1}" />
                </div>
              ` : ""}
              <div style="display:flex; justify-content:flex-end; gap:0.5rem; margin-top:1.2rem;">
                <button type="button" class="gb-btn gb-btn-outline" id="gb-modal-cancel">Cancel</button>
                <button type="submit" class="gb-btn gb-btn-primary">Save Changes</button>
              </div>
            </form>
          </div>
        </div>
      `;

      const closeModal = () => { modalHost.innerHTML = ""; };
      modalHost.querySelector("#gb-modal-close").addEventListener("click", closeModal);
      modalHost.querySelector("#gb-modal-cancel").addEventListener("click", closeModal);

      modalHost.querySelector("#gb-edit-form").addEventListener("submit", (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        dev.name = formData.get("name");
        dev.bus = formData.get("bus");
        if (formData.has("capacity_mwh")) dev.capacity_mwh = parseFloat(formData.get("capacity_mwh"));
        if (formData.has("max_p_mw")) dev.max_p_mw = parseFloat(formData.get("max_p_mw"));
        if (formData.has("min_p_mw")) dev.min_p_mw = parseFloat(formData.get("min_p_mw"));
        if (formData.has("efficiency")) dev.efficiency = parseFloat(formData.get("efficiency"));
        if (formData.has("p_mw")) dev.p_mw = parseFloat(formData.get("p_mw"));
        if (formData.has("q_mvar")) dev.q_mvar = parseFloat(formData.get("q_mvar"));

        closeModal();
        this.render();
      });
    }

    generatePythonCode() {
      const preset = PRESETS[this.currentPresetKey];
      let code = `import gymnasium as gym\n`;
      code += `import numpy as np\n`;
      code += `import gridages\n`;
      code += `from gridages.devices import ESS, DG, RES, Grid\n`;
      code += `from gridages.envs.single_agent import ${preset.envClass}\n\n`;

      code += `# 1. Define Microgrid DER Devices\n`;
      code += `devices = [\n`;

      this.devices.forEach(dev => {
        if (dev.type === "battery") {
          code += `    ESS(\n`;
          code += `        name="${dev.name}",\n`;
          code += `        bus="${dev.bus}",\n`;
          code += `        capacity_mwh=${dev.capacity_mwh || 1.0},\n`;
          code += `        max_p_mw=${dev.max_p_mw || 0.5},\n`;
          code += `        efficiency=${dev.efficiency || 0.95},\n`;
          code += `    ),\n`;
        } else if (dev.type === "solar") {
          code += `    RES(\n`;
          code += `        name="${dev.name}",\n`;
          code += `        bus="${dev.bus}",\n`;
          code += `        max_p_mw=${dev.max_p_mw || 0.6},\n`;
          code += `        type="solar",\n`;
          code += `    ),\n`;
        } else if (dev.type === "wind") {
          code += `    RES(\n`;
          code += `        name="${dev.name}",\n`;
          code += `        bus="${dev.bus}",\n`;
          code += `        max_p_mw=${dev.max_p_mw || 1.2},\n`;
          code += `        type="wind",\n`;
          code += `    ),\n`;
        } else if (dev.type === "generator") {
          code += `    DG(\n`;
          code += `        name="${dev.name}",\n`;
          code += `        bus="${dev.bus}",\n`;
          code += `        min_p_mw=${dev.min_p_mw || 0.1},\n`;
          code += `        max_p_mw=${dev.max_p_mw || 0.8},\n`;
          code += `        cost_curve_coefs=[0.02, 25.0, 5.0],\n`;
          code += `    ),\n`;
        } else if (dev.type === "load") {
          code += `    # Load attached at Bus ${dev.bus}: P=${dev.p_mw || 0.4} MW, Q=${dev.q_mvar || 0.1} MVar\n`;
        }
      });

      code += `]\n\n`;
      code += `# 2. Instantiate Environment with configured DERs\n`;
      code += `env = ${preset.envClass}(devices=devices)\n\n`;
      code += `# 3. Reset and Run RL Step\n`;
      code += `obs, info = env.reset(seed=42)\n`;
      code += `action = env.action_space.sample()  # Example action\n`;
      code += `obs, reward, terminated, truncated, info = env.step(action)\n`;
      code += `print(f"Step Reward: {reward:.4f}, Net Load: {info.get('net_load')}")\n`;

      return code;
    }
  }

  // Auto-initialize builder on any container with class .gridages-interactive-builder
  document.addEventListener("DOMContentLoaded", () => {
    const containers = document.querySelectorAll(".gridages-interactive-builder");
    containers.forEach(elem => new GridBuilderApp(elem));
  });
})();
