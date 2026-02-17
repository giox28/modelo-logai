import { Component, EventEmitter, Output, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { PlotlyService } from '../../services/plotly.service';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css'
})
export class UploadComponent implements OnInit {
  @Output() processingComplete = new EventEmitter<any>();

  selectedBasin: string = 'guajira';
  file: File | null = null;
  fileName: string = '';
  isDragOver: boolean = false;

  curvesInFile: string[] = [];
  availableModels: string[] = [];

  // Object to track checkbox state: { 'DT': true, 'RHOB': false }
  targetSelection: { [key: string]: boolean } = {};

  isLoading: boolean = false;

  constructor(
    private apiService: ApiService,
    private plotlyService: PlotlyService
  ) { }

  ngOnInit() {
    // Pre-fetch models to ensure connectivity
    this.fetchModels();
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = false;
    if (event.dataTransfer?.files.length) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  onFileSelected(event: any) {
    if (event.target.files.length) {
      this.handleFile(event.target.files[0]);
    }
  }

  // Input Curve Selection
  inputCurveSelection: { [key: string]: boolean } = {};

  get inputCurvesList(): string[] {
    return this.curvesInFile;
  }

  toggleInputCurve(curve: string) {
    this.inputCurveSelection[curve] = !this.inputCurveSelection[curve];
    this.updateInputPlot();
  }

  // Store inspected data to re-render without API call
  lastInspectedData: any = null;

  async handleFile(file: File) {
    this.file = file;
    this.fileName = file.name;
    this.isLoading = true; // Show loading while inspecting

    // 1. Inspect File
    this.apiService.inspectFile(file).subscribe({
      next: (data) => {
        this.isLoading = false;
        if (data.curves) {
          this.lastInspectedData = data;
          this.curvesInFile = Object.keys(data.curves);

          // Initialize Input Selection (Default: All active)
          this.curvesInFile.forEach(c => this.inputCurveSelection[c] = true);

          this.updateInputPlot();
          this.fetchModels();
        }
      },
      error: (err) => {
        this.isLoading = false;
        console.error('Error inspecting file', err);
        alert('Error inspeccionando archivo: ' + err.message);
      }
    });
  }

  updateInputPlot() {
    if (!this.lastInspectedData) return;

    const visibleCurves: any = {};
    for (const curve of this.curvesInFile) {
      if (this.inputCurveSelection[curve]) {
        visibleCurves[curve] = this.lastInspectedData.curves[curve];
      }
    }

    this.renderInputPlot(this.lastInspectedData.depth, visibleCurves);
  }

  onBasinChange() {
    if (this.file) {
      this.fetchModels();
    }
  }

  fetchModels() {
    this.apiService.getAvailableModels(this.selectedBasin).subscribe({
      next: (data) => {
        console.log('Models fetched:', data); // Debug
        this.availableModels = data.models || [];
        this.initTargetSelection();
      },
      error: (err) => console.error('Error fetching models', err)
    });
  }

  initTargetSelection() {
    this.targetSelection = {};
    this.availableModels.forEach(model => {
      // Default check if NOT in file (Reconstruction candidate)
      const exists = this.curvesInFile.includes(model);
      this.targetSelection[model] = !exists;
    });
  }

  isCurveInFile(curve: string): boolean {
    return this.curvesInFile.includes(curve);
  }

  get selectedTargetsList(): string[] {
    return Object.keys(this.targetSelection).filter(k => this.targetSelection[k]);
  }

  getCurveConfig(curveName: string): any {
    const name = curveName.toUpperCase();
    if (name.includes('ILD') || name.includes('RES') || name.includes('LLD')) return { type: 'log', title: 'Ohm.m' };
    if (name.includes('DT') || name.includes('SONIC')) return { autorange: 'reversed', title: 'us/ft' };
    if (name.includes('NPHI') || name.includes('PHIE') || name.includes('POR')) return { range: [0.6, -0.1], title: 'v/v' };
    if (name.includes('GR') || name.includes('GAMMA')) return { range: [0, 200], title: 'API' };
    if (name.includes('RHOB') || name.includes('DEN')) return { range: [1.95, 2.95], title: 'g/cc' };
    return { autorange: true, title: 'Valor' };
  }

  renderInputPlot(depth: number[], curves: any) {
    const traces: any[] = [];
    const layout: any = {
      title: 'Vista Previa (Tracks Independientes)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#94a3b8' },
      yaxis: { title: 'Profundidad', autorange: 'reversed' },
      height: 500,
      margin: { t: 80, r: 20, b: 40, l: 60 },
      showlegend: false
    };

    // Filter curves based on data passed (which is already filtered by updateInputPlot)
    const activeCurves = Object.entries(curves).filter(([, d]) => d && (d as any[]).length > 0);
    const count = activeCurves.length;

    if (count === 0) {
      this.plotlyService.newPlot('inputPlot', [], layout);
      return;
    }

    const gap = 0.05;
    const width = (1 - (count - 1) * gap) / count;

    activeCurves.forEach(([name, data], i) => {
      const axisName = i === 0 ? 'xaxis' : 'xaxis' + (i + 1);
      const axisId = i === 0 ? 'x' : 'x' + (i + 1); // FIXED: 'x' instead of 'x1'
      const config = this.getCurveConfig(name);

      // Calculate Domain
      const start = i * (width + gap);
      const end = start + width;

      layout[axisName] = {
        title: `${name} (${config.title})`,
        type: config.type || 'linear',
        range: config.range,
        autorange: config.autorange,
        side: 'top',
        anchor: 'free', // Important for independent movement
        position: 1,    // Keep title at top
        domain: [start, end], // EXPLICIT SEPARATION
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        zeroline: false,
        showline: true, // Add border to track
        mirror: true   // Add border to track
      };

      traces.push({
        y: depth,
        x: data,
        name: name,
        type: 'scatter',
        mode: 'lines',
        line: { width: 1 },
        xaxis: axisId
      });
    });

    this.plotlyService.newPlot('inputPlot', traces, layout);
  }

  onSubmit() {
    if (!this.file) return;

    this.isLoading = true;

    const formData = new FormData();
    formData.append('file', this.file);
    formData.append('basin_name', this.selectedBasin);

    const targets = this.selectedTargetsList.join(',');
    formData.append('target_curves', targets);

    this.apiService.processWell(formData).subscribe({
      next: (data) => {
        this.isLoading = false;
        this.processingComplete.emit(data);
      },
      error: (err) => {
        this.isLoading = false;
        alert('Error procesando archivo: ' + err.message);
      }
    });
  }
}
