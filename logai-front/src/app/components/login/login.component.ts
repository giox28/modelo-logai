import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
    selector: 'app-login',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
    <div class="login-container">
      <div class="login-box card">
        <h2 style="color:var(--text-main); text-align:center; margin-bottom: 5px;">GeoOptima</h2>
        <p style="color:var(--text-muted); text-align:center; margin-bottom: 25px;">Acceso Seguro a la Plataforma</p>
        
        <form (ngSubmit)="onSubmit()" #loginForm="ngForm">
          <div class="form-group" style="margin-bottom: 15px;">
            <label for="username" style="color:var(--text-main); display:block; margin-bottom:5px;">Usuario</label>
            <input type="text" id="username" name="username" [(ngModel)]="username" required
                   style="width: 100%; padding: 10px; background: rgba(0,0,0,0.5); border: 1px solid var(--glass-border); color: white; border-radius: 6px;">
          </div>
          <div class="form-group" style="margin-bottom: 25px;">
            <label for="password" style="color:var(--text-main); display:block; margin-bottom:5px;">Contraseña</label>
            <input type="password" id="password" name="password" [(ngModel)]="password" required
                   style="width: 100%; padding: 10px; background: rgba(0,0,0,0.5); border: 1px solid var(--glass-border); color: white; border-radius: 6px;">
          </div>
          
          <div *ngIf="error" style="color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 10px; border-radius: 4px; margin-bottom: 15px; text-align: center;">
            {{error}}
          </div>

          <button type="submit" [disabled]="loading || !loginForm.form.valid"
                  style="width: 100%; padding: 12px; background: var(--secondary); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s;">
            {{ loading ? 'Autenticando...' : 'Iniciar Sesión' }}
          </button>
        </form>
      </div>
    </div>
  `,
    styles: [`
    .login-container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      width: 100%;
      background: radial-gradient(circle at top right, rgba(168, 85, 247, 0.05) 0%, transparent 40%),
                radial-gradient(circle at bottom left, rgba(236, 72, 153, 0.05) 0%, transparent 40%);
    }
    .login-box {
      width: 100%;
      max-width: 400px;
      padding: 40px;
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    button:disabled { opacity: 0.7; cursor: not-allowed; }
    button:hover:not(:disabled) { filter: brightness(1.1); }
  `]
})
export class LoginComponent {
    username = '';
    password = '';
    loading = false;
    error = '';

    constructor(
        private authService: AuthService,
        private router: Router,
        private route: ActivatedRoute
    ) { }

    onSubmit() {
        this.loading = true;
        this.error = '';

        this.authService.login(this.username, this.password).subscribe({
            next: (success) => {
                if (success) {
                    const returnUrl = this.route.snapshot.queryParams['returnUrl'] || '/';
                    this.router.navigateByUrl(returnUrl);
                } else {
                    this.error = 'Error inesperado';
                    this.loading = false;
                }
            },
            error: (err) => {
                this.error = 'Usuario o contraseña incorrectos';
                this.loading = false;
                console.error(err);
            }
        });
    }
}
