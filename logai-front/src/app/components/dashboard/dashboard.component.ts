import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UploadComponent } from '../upload/upload.component';
import { ResultsComponent } from '../results/results.component';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, UploadComponent, ResultsComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent {
  constructor(private router: Router, private authService: AuthService) { }

  resultData: any = null;

  onProcessingComplete(data: any) {
    this.resultData = data;
    // Scroll to results
    setTimeout(() => {
      const el = document.getElementById('results-section');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  }

  logout() {
    this.authService.logout();
    this.router.navigate(['/login']);
  }
}
