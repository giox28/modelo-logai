import { Injectable } from '@angular/core';

declare const Plotly: any;

@Injectable({
  providedIn: 'root'
})
export class PlotlyService {

  constructor() { }

  newPlot(divId: string, data: any[], layout: any): void {
    if (typeof Plotly !== 'undefined') {
      Plotly.newPlot(divId, data, layout);
    } else {
      console.error('Plotly is not loaded');
    }
  }

  react(divId: string, data: any[], layout: any): void {
    if (typeof Plotly !== 'undefined') {
      Plotly.react(divId, data, layout);
    } else {
      console.error('Plotly is not loaded');
    }
  }
}
