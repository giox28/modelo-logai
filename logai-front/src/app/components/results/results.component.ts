import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PlotlyService } from '../../services/plotly.service';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results.component.html',
  styleUrl: './results.component.css'
})
export class ResultsComponent implements OnChanges {
  @Input() data: any; // data from /process-well

  constructor(private plotlyService: PlotlyService) { }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && this.data) {
      // Allow DOM to update first
      setTimeout(() => {
        this.renderResultPlot();
      }, 100);
    }
  }

  get voiReportItems(): any[] {
    if (!this.data || !this.data.voi_report) return [];
    // Convert voi_report object to array
    return Object.entries(this.data.voi_report)
      .filter(([key]) => key !== 'error')
      .map(([target, info]) => ({ target, info }));
  }

  getFeatureList(info: any): any[] {
    if (!info || !info.importance) return [];
    return Object.entries(info.importance)
      .filter(([, val]: [string, any]) => val > 0.01)
      .map(([feat, val]) => ({ feat, percent: (val as number * 100).toFixed(1) }));
  }

  getCurveConfig(curveName: string): any {
    const name = curveName.toUpperCase();

    // Configuración Geocientífica de Escalas
    if (name.includes('ILD') || name.includes('RES') || name.includes('LLD')) {
      return { type: 'log', title: 'Ohm.m' };
    }
    if (name.includes('DT') || name.includes('SONIC')) {
      return { autorange: 'reversed', title: 'us/ft' };
    }
    if (name.includes('NPHI') || name.includes('PHIE') || name.includes('POR')) {
      return { range: [0.6, -0.1], title: 'v/v' }; // Porosidad inversa estándar
    }
    if (name.includes('GR') || name.includes('GAMMA')) {
      return { range: [0, 200], title: 'API' };
    }
    if (name.includes('RHOB') || name.includes('DEN')) {
      return { range: [1.95, 2.95], title: 'g/cc' };
    }

    return { autorange: true, title: 'Valor' };
  }

  renderResultPlot() {
    console.log('ResultsComponent: renderResultPlot called');
    console.log('ResultsComponent: data:', this.data);

    if (!this.data || !this.data.depth_data || !this.data.synthetic_data) {
      console.error('ResultsComponent: Missing data for plot');
      return;
    }

    console.log('ResultsComponent: synthetic_data keys:', Object.keys(this.data.synthetic_data));


    const traces: any[] = [];
    const layout: any = {
      title: 'Curvas Reconstruidas (Tracks Independientes)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#94a3b8' },
      yaxis: { title: 'Profundidad', autorange: 'reversed' },
      height: 600,
      margin: { t: 100, r: 20, b: 40, l: 60 },
      showlegend: false
    };

    const depth = this.data.depth_data;
    const activeCurves = Object.entries(this.data.synthetic_data).filter(([, d]) => d);
    const count = activeCurves.length;

    if (count === 0) return;

    const gap = 0.05;
    const width = (1 - (count - 1) * gap) / count;

    activeCurves.forEach(([target, curveData], i) => {
      const axisName = i === 0 ? 'xaxis' : 'xaxis' + (i + 1);
      const axisId = i === 0 ? 'x' : 'x' + (i + 1); // FIXED: 'x' instead of 'x1'
      const config = this.getCurveConfig(target);

      const start = i * (width + gap);
      const end = start + width;

      layout[axisName] = {
        title: `${target} (${config.title})`,
        type: config.type || 'linear',
        range: config.range,
        autorange: config.autorange,
        side: 'top',
        anchor: 'free',
        position: 1,
        domain: [start, end],
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        zeroline: false,
        showline: true,
        mirror: true
      };

      // Handle v2.0 Data Structure (Object with P10/P50/P90)
      const dataObj = curveData as any;
      const xP50 = dataObj.P50_SYN;
      const xP10 = dataObj.P10;
      const xP90 = dataObj.P90;

      // 1. P10 Trace (Lower Bound) - Invisible
      if (xP10 && xP10.length > 0) {
        traces.push({
          y: depth,
          x: xP10,
          type: 'scatter',
          mode: 'lines',
          line: { width: 0 },
          xaxis: axisId,
          showlegend: false,
          hoverinfo: 'skip'
        });
      }

      // 2. P90 Trace (Upper Bound) - Filled to P10
      if (xP90 && xP90.length > 0) {
        traces.push({
          y: depth,
          x: xP90,
          type: 'scatter',
          mode: 'lines',
          line: { width: 0 },
          xaxis: axisId,
          fill: 'tonextx',
          fillcolor: 'rgba(34, 197, 94, 0.2)', // Green tint
          showlegend: false,
          hoverinfo: 'skip'
        });
      }

      // 3. Main P50 Trace
      traces.push({
        y: depth,
        x: xP50,
        name: `${target}`,
        type: 'scatter',
        mode: 'lines',
        line: { width: 1.5, color: '#22c55e' },
        xaxis: axisId
      });
    });

    this.plotlyService.newPlot('resultPlot', traces, layout);
  }
}
