"""Differentiable spectrogram modules for Birds Distillation Edge."""

import torch
import torchaudio
import logging
import torchaudio.functional as F
# import matplotlib.pyplot as plt  # Commented out - only used in commented test code
import torch.nn as nn
import numpy as np
import math # Per pi, cos, etc. se necessario per filtri più avanzati

logger = logging.getLogger(__name__)

# Load audio file
def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    # Convert to mono if stereo
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(), sr

# Parameters
n_filters = 40  # Number of filters
f_min = 1000  # Minimum frequency in Hz
f_max = 8000  # Maximum frequency in Hz
n_fft = 512  # FFT size
hop_length = 128  # Hop length for STFT
breakpoint = 500  # Frequency where we switch from log to linear
transition_width = 300  # Smooth transition width for differentiable log-linear

def compute_stft(waveform, n_fft, hop_length):
    """
    Compute STFT magnitude spectrogram
    
    Args:
        waveform: [n_samples]
        n_fft: FFT size
        hop_length: Hop size
    
    Returns:
        magnitude: [n_freq, n_time]
    """
    # Ensure waveform is 1D
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    if waveform.dim() == 0:
        waveform = waveform.unsqueeze(0)
        
    stft = torch.stft(waveform, 
                     n_fft=n_fft, 
                     hop_length=hop_length,
                     window=torch.hann_window(n_fft).to(waveform.device),
                     return_complex=True)
    
    # Make sure the output is 2D [n_freq, n_time]
    return torch.abs(stft)

def get_mel_filter_bank(sr, n_fft, n_filters, f_min, f_max):
    """
    Create mel filter bank
    
    Args:
        sr: sample rate
        n_fft: FFT size
        n_filters: number of mel bands
        f_min: minimum frequency
        f_max: maximum frequency
    
    Returns:
        filter_bank: [n_filters, n_freq] where n_freq = n_fft//2 + 1
    """
    # Get the mel filter bank
    filter_bank = F.melscale_fbanks(
        n_freqs=n_fft//2 + 1,  # This must match STFT output
        f_min=f_min,
        f_max=f_max,
        n_mels=n_filters,
        sample_rate=sr,
        norm='slaney',
        mel_scale='htk'
    )
    
    # Transpose if needed to ensure shape is [n_filters, n_freq]
    if filter_bank.shape[0] != n_filters:
        filter_bank = filter_bank.T
        
    return filter_bank

# Compute log-spaced center frequencies
def get_log_filter_bank(n_filters, f_min, f_max, n_fft, sr):
    log_center_freqs = torch.logspace(torch.log10(torch.tensor(f_min)), 
                                      torch.log10(torch.tensor(f_max)), 
                                      steps=n_filters)
    log_fft_bins = torch.round((log_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    log_fft_bins = torch.clamp(log_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(log_fft_bins, n_filters, n_fft)

# Compute piecewise log-linear center frequencies
def get_piecewise_filter_bank(n_filters, f_min, f_max, breakpoint, n_fft, sr):
    n_low = n_filters // 2
    n_high = n_filters - n_low
    low_freqs = torch.logspace(torch.log10(torch.tensor(f_min)), 
                               torch.log10(torch.tensor(breakpoint)), 
                               steps=n_low)
    high_freqs = torch.linspace(breakpoint, f_max, steps=n_high)
    piecewise_center_freqs = torch.cat([low_freqs, high_freqs])
    piecewise_fft_bins = torch.round((piecewise_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    piecewise_fft_bins = torch.clamp(piecewise_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(piecewise_fft_bins, n_filters, n_fft)

# Compute differentiable piecewise log-linear center frequencies
def get_differentiable_filter_bank(n_filters, f_min, f_max, breakpoint, transition_width, n_fft, sr):
    x = torch.linspace(0, 1, steps=n_filters)
    log_part = f_min * (f_max / f_min) ** x
    linear_part = f_min + x * (f_max - f_min)
    S = torch.sigmoid((x - (breakpoint - f_min) / (f_max - f_min)) * transition_width)
    differentiable_center_freqs = (1 - S) * log_part + S * linear_part
    differentiable_fft_bins = torch.round((differentiable_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    differentiable_fft_bins = torch.clamp(differentiable_fft_bins, 1, (n_fft // 2) - 2)
    return generate_filter_bank(differentiable_fft_bins, n_filters, n_fft)

# Function to generate triangular filters
def generate_filter_bank(fft_bins, n_filters, n_fft):
    filter_bank = torch.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left = fft_bins[i - 1] if i > 0 else 0
        center = fft_bins[i]
        right = fft_bins[i + 1] if i < n_filters - 1 else (n_fft // 2)

        if center > left and center - left > 1:
            filter_bank[i, left:center] = torch.linspace(0, 1, steps=center - left)
        if right > center and right - center > 1:
            filter_bank[i, center:right] = torch.linspace(1, 0, steps=right - center)
    return filter_bank

def apply_filter_bank(filter_bank, magnitude):
    """
    Apply filter bank to STFT magnitude spectrogram
    
    Args:
        filter_bank: [n_filters, n_freq]
        magnitude: [n_freq, n_time]
    
    Returns:
        filtered: [n_filters, n_time]
    """
    # Ensure both tensors are on same device and dtype
    filter_bank = filter_bank.to(magnitude.device).float()
    magnitude = magnitude.float()
    
    # Check dimensions
    assert filter_bank.shape[1] == magnitude.shape[0], \
        f"Frequency dimension mismatch: filter_bank {filter_bank.shape} vs magnitude {magnitude.shape}"
    
    # [n_filters, n_freq] @ [n_freq, n_time] = [n_filters, n_time]
    return torch.matmul(filter_bank, magnitude)

# Function to convert to dB scale and normalize
def to_db_scale(S, ref=None):
    if ref is None:
        ref = torch.max(S)
    return 20 * torch.log10(torch.clamp(S, min=1e-10) / ref)

# Differentiability check function
def check_differentiability():
    # Test parameters
    n_filters = 10
    f_min = 500.0
    f_max = 8000.0
    breakpoint = 1000.0
    transition_width = 50.0
    n_fft = 512
    sr = 22050
    
    # Create input with requires_grad
    x = torch.linspace(0, 1, steps=n_filters, requires_grad=True)
    
    # Compute center frequencies
    log_part = f_min * (f_max / f_min) ** x
    linear_part = f_min + x * (f_max - f_min)
    S = torch.sigmoid((x - (breakpoint - f_min) / (f_max - f_min)) * transition_width)
    freqs = (1 - S) * log_part + S * linear_part
    
    # Test backpropagation
    try:
        loss = freqs.sum()
        loss.backward()
        if x.grad is not None:
            logger.info("Differentiability check PASSED")
            logger.info("Sample gradients:", x.grad[:3])
            return True
        else:
            logger.warning("Differentiability check FAILED (no gradients)")
            return False
    except Exception as e:
        logger.error("Differentiability check FAILED: %s", e)
        return False

def get_differentiable_filter_bank(n_filters, f_min, f_max, breakpoint, transition_width, n_fft, sr):
    # Create parameter space
    x = torch.linspace(0, 1, steps=n_filters)
    
    # Compute in log domain directly
    log_min = torch.log(torch.tensor(f_min))
    log_max = torch.log(torch.tensor(f_max))
    log_break = torch.log(torch.tensor(breakpoint))
    
    # Compute log-spaced frequencies
    log_freqs = torch.exp(log_min + x * (log_max - log_min))
    
    # Compute linear-spaced frequencies
    lin_freqs = f_min + x * (f_max - f_min)
    
    # Create a smoother transition using sigmoid
    normalized_break = (log_break - log_min) / (log_max - log_min)
    S = torch.sigmoid((x - normalized_break) * transition_width)
    
    # Combine log and linear parts
    differentiable_center_freqs = (1 - S) * log_freqs + S * lin_freqs
    
    # Convert to FFT bins
    differentiable_fft_bins = torch.round((differentiable_center_freqs / (sr / 2)) * (n_fft // 2)).long()
    differentiable_fft_bins = torch.clamp(differentiable_fft_bins, 1, (n_fft // 2) - 2)
    
    return generate_filter_bank(differentiable_fft_bins, n_filters, n_fft)

# Revised differentiability check function
def check_differentiability():
    # Test parameters
    n_filters = 10
    f_min = 500.0
    f_max = 8000.0
    breakpoint = 1000.0
    transition_width = 10.0  # Reduced from 50.0 to decrease gradient magnitude
    n_fft = 512
    sr = 22050
    
    # Create input with requires_grad
    x = torch.linspace(0, 1, steps=n_filters, requires_grad=True)
    
    # Compute center frequencies using a more stable approach
    # Using log domain directly
    log_min = torch.tensor(torch.log(torch.tensor(f_min)))
    log_max = torch.tensor(torch.log(torch.tensor(f_max)))
    log_break = torch.tensor(torch.log(torch.tensor(breakpoint)))
    
    # Compute log-spaced frequencies
    log_freqs = torch.exp(log_min + x * (log_max - log_min))
    
    # Compute linear-spaced frequencies
    lin_freqs = f_min + x * (f_max - f_min)
    
    # Create a smoother transition using sigmoid
    normalized_break = (torch.log(torch.tensor(breakpoint)) - log_min) / (log_max - log_min)
    S = torch.sigmoid((x - normalized_break) * transition_width)
    
    # Combine log and linear parts
    freqs = (1 - S) * log_freqs + S * lin_freqs
    
    # Apply scaling factor to reduce gradient magnitude
    scaling_factor = 0.01
    freqs_scaled = freqs * scaling_factor
    
    # Test backpropagation
    try:
        loss = freqs_scaled.sum()
        loss.backward()
        if x.grad is not None:
            logger.info("Differentiability check PASSED")
            logger.info("Sample gradients:", x.grad[:3])
            return True
        else:
            logger.warning("Differentiability check FAILED (no gradients)")
            return False
    except Exception as e:
        logger.error("Differentiability check FAILED: %s", e)
        return False

class FullyLearnableFilterBank(nn.Module):
    """Completely learnable filter bank - ogni coefficiente è un parametro"""
    
    def __init__(self, n_filters=64, n_freq_bins=513, sample_rate=24000, init_strategy='triangular_noise', n_fft=1024, hop_length=240):
        super().__init__()
        self.n_filters = n_filters
        self.n_freq_bins = n_freq_bins
        self.sample_rate = sample_rate
        self.init_strategy = init_strategy
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # La matrice di filtri completamente apprendibile
        self.filter_bank = nn.Parameter(
            self._initialize_filters_intelligently(n_filters, n_freq_bins)
        )
        
    def forward(self, waveform):
        """
        Args:
            waveform: [batch, time] or [time] - raw audio waveform
        Returns:
            filtered_features: [batch, n_filters, time_frames]
        """
        # Handle single waveform input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        device = waveform.device
        
        # 1. Compute STFT
        window = torch.hann_window(self.n_fft, device=device)
        stft_complex = torch.stft(waveform, 
                                  n_fft=self.n_fft, 
                                  hop_length=self.hop_length,
                                  window=window,
                                  return_complex=True)  # [batch, freq_bins, time_frames]
        
        magnitude_stft = torch.abs(stft_complex)  # [batch, freq_bins, time_frames]
        
        # 2. Apply learnable filter bank
        # filter_bank: [n_filters, freq_bins] = [n, f]
        # magnitude_stft: [batch, freq_bins, time_frames] = [b, f, t]
        # Result: [batch, n_filters, time_frames] = [b, n, t]
        return torch.einsum('nf,bft->bnt', self.filter_bank, magnitude_stft)
    
    def _initialize_filters_intelligently(self, n_filters, n_freq_bins):
        """Diverse strategie di inizializzazione"""
        
        if self.init_strategy == 'random':
            # Semplice random Gaussian
            return torch.randn(n_filters, n_freq_bins) * 0.01
            
        elif self.init_strategy == 'triangular_noise':
            # Inizializza come filtri triangolari + noise
            triangular_base = self._generate_triangular_filters(n_filters, n_freq_bins)
            noise = torch.randn_like(triangular_base) * 0.01
            return triangular_base + noise
            
        elif self.init_strategy == 'xavier':
            # Xavier initialization
            std = np.sqrt(2.0 / (n_filters + n_freq_bins))
            return torch.randn(n_filters, n_freq_bins) * std
            
        else:
            raise ValueError(f"Unknown init strategy: {self.init_strategy}")
    
    def _generate_triangular_filters(self, n_filters, n_freq_bins):
        """Genera baseline triangolare come starting point"""
        # Create triangular filters as baseline
        filter_bank = torch.zeros(n_filters, n_freq_bins)
        
        # Compute center frequencies (log-spaced for better coverage)
        f_min = 150.0
        f_max = self.sample_rate / 2.0  # Nyquist frequency
        
        # Log-spaced center frequencies
        log_min = np.log(f_min)
        log_max = np.log(f_max)
        center_freqs = torch.exp(torch.linspace(log_min, log_max, n_filters))
        
        # Convert frequencies to bin indices
        freq_to_bin = (n_freq_bins - 1) / (self.sample_rate / 2.0)
        center_bins = (center_freqs * freq_to_bin).long()
        center_bins = torch.clamp(center_bins, 1, n_freq_bins - 2)
        
        # Generate triangular filters
        for i in range(n_filters):
            left = center_bins[i - 1].item() if i > 0 else 0
            center = center_bins[i].item()
            right = center_bins[i + 1].item() if i < n_filters - 1 else n_freq_bins - 1
            
            # Left slope
            if center > left:
                slope_len = center - left
                if slope_len > 0:
                    filter_bank[i, left:center] = torch.linspace(0, 1, slope_len)
            
            # Right slope
            if right > center:
                slope_len = right - center
                if slope_len > 0:
                    filter_bank[i, center:right] = torch.linspace(1, 0, slope_len)
        
        return filter_bank


class DifferentiableSpectrogram(nn.Module):
    def __init__(self, sr=24000, n_filters=64,
                 f_min=150.0, f_max=10000.0, 
                 n_fft=1024, hop_length=240,
                 initial_breakpoint=4000.0, 
                 initial_transition_width=100.0,
                 trainable_filterbank=True, # Flag per controllare l'apprendimento dei parametri del filtro
                 spec_type="combined_log_linear", # Aggiunto per selezionare il tipo di filtro
                 debug=False):
        super().__init__()
        self.sr = sr
        self.n_filters = n_filters
        self.f_min_val = float(f_min) # Memorizza come float per usi non-tensoriali
        self.f_max_val = float(f_max)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trainable_filterbank = trainable_filterbank
        self.spec_type = spec_type
        self.debug = debug

        # Rendi breakpoint e transition_width parametri apprendibili se richiesto
        if self.trainable_filterbank and self.spec_type == "combined_log_linear":
            # Inizializza come scalari, ma nn.Parameter li renderà tensori 0-dim
            self.breakpoint = nn.Parameter(torch.tensor(float(initial_breakpoint)))
            self.transition_width = nn.Parameter(torch.tensor(float(initial_transition_width)))
            # Si potrebbe aggiungere un clamp o una trasformazione (es. softplus) per transition_width 
            # se si vuole che rimanga positivo o in un certo range.
            # Per breakpoint, potrebbe essere necessario un clamp all'intervallo [f_min, f_max]
        else:
            # Mantienili come attributi fissi (float) se non devono essere addestrati o se il tipo di spec è diverso
            # Registra come buffer se vuoi che siano nello state_dict ma non apprendibili
            self.register_buffer('breakpoint', torch.tensor(float(initial_breakpoint)))
            self.register_buffer('transition_width', torch.tensor(float(initial_transition_width)))
        
        # Registra f_min e f_max come buffer in modo che siano nello state_dict e si spostino con il modello (es. to(device))
        self.register_buffer('f_min', torch.tensor(float(f_min)))
        self.register_buffer('f_max', torch.tensor(float(f_max)))

        # Aggiungi cache per la filter bank
        self._filter_bank_cache = None
        self._cached_params = {}

    def rebuild_filters(self):
        """
        Invalida la cache della filter bank, forzando il ricalcolo al prossimo forward pass.
        Questo è il metodo da chiamare quando i parametri vengono modificati esternamente.
        """
        self._filter_bank_cache = None
        self._cached_params = {}
        if self.debug:
            logger.debug("Spec: Filter cache invalidato.")


    def _get_triangular_filter_centers(self, device):
        """
        Calcola le frequenze centrali per i filtri triangolari.
        Questo metodo determina le posizioni dei centri dei filtri sulla scala delle frequenze.
        """
        if self.spec_type == "mel":
            # Per Mel, le frequenze centrali sono implicitamente definite da melscale_fbanks
            # Questa funzione restituisce direttamente la filter bank, non solo i centri.
            # Quindi, se spec_type == "mel", _get_current_filter_bank gestirà questo.
            # Qui, potremmo restituire None o placeholder se questa funzione è chiamata genericamente.
            # Per semplicità, la logica Mel sarà in _get_current_filter_bank.
            return None 
            
        elif self.spec_type == "linear_triangular":
            # Spaziatura lineare dei centri dei filtri
            return torch.linspace(self.f_min.item(), self.f_max.item(), self.n_filters, device=device)

        elif self.spec_type == "combined_log_linear":
            # x definisce la posizione normalizzata di ogni filtro [0, 1]
            x = torch.linspace(0, 1, steps=self.n_filters, device=device)

            # Usa i valori correnti di breakpoint e transition_width (che sono nn.Parameter se trainable)
            # f_min e f_max sono buffer, quindi già tensori.
            current_breakpoint = self.breakpoint 
            current_transition_width = self.transition_width
            
            # Allow breakpoint to go to any value - no artificial constraints
            # The system can naturally find the optimal breakpoint, even if very low
            # Low breakpoint values (near 0) simply mean "prefer linear representation"
            clamped_breakpoint = current_breakpoint  # No clamping - let the gradients decide!

            log_part = self.f_min * (self.f_max / self.f_min) ** x
            linear_part = self.f_min + x * (self.f_max - self.f_min)
            
            f_range = self.f_max - self.f_min
            # Evita divisione per zero se f_range è troppo piccolo (improbabile con clamp di breakpoint)
            if torch.abs(f_range) < 1e-6:
                normalized_breakpoint_pos = torch.full_like(x, 0.5) # Default a metà
            else:
                normalized_breakpoint_pos = (clamped_breakpoint - self.f_min) / f_range
            
            # Sigmoid per la transizione soft
            # Aggiungi un clamp a transition_width se necessario, es. > 0
            # Un transition_width molto grande rende S quasi uno step function.
            # Un transition_width piccolo la rende molto smooth.
            # Si potrebbe usare softplus per assicurare che transition_width > 0
            # current_transition_width = F.softplus(self.transition_width) # Esempio
            
            S = torch.sigmoid((x - normalized_breakpoint_pos) * current_transition_width)
            
            center_freqs = (1 - S) * log_part + S * linear_part
            
            # Assicura che le frequenze centrali siano monotonamente crescenti e all'interno [f_min, f_max]
            # Il clamp precedente su breakpoint aiuta, ma la combinazione potrebbe uscire.
            # Un sort + clamp finale potrebbe essere necessario se l'apprendimento le fa divergere troppo.
            # Per ora, facciamo un clamp finale.
            center_freqs = torch.clamp(center_freqs, self.f_min.item(), self.f_max.item())
            
            # Opzionale: forza monotonicità (questo potrebbe interrompere i gradienti se fatto in modo hard)
            # if self.trainable_filterbank: # Solo se apprendibili, altrimenti dovrebbero essere già ok
            #     sorted_freqs, _ = torch.sort(center_freqs)
            #     if not torch.allclose(center_freqs, sorted_freqs) and self.debug:
            #         print(f"DEBUG Spec: Center freqs non-monotonic. Before sort: {center_freqs.detach().cpu().numpy().round(1)}")
            #         # center_freqs = sorted_freqs # Sostituire interrompe i gradienti diretti
            #         # Invece, si potrebbe aggiungere un piccolo termine di loss che penalizzi la non-monotonicità.
            
            return center_freqs
        else:
            raise ValueError(f"Unsupported spec_type: {self.spec_type}")


    def _generate_triangular_filter_bank_from_centers(self, center_freqs, device):
        """
        Genera una filter bank triangolare differenziabile date le frequenze centrali.
        `center_freqs` sono le posizioni centrali dei filtri (potrebbero essere frazionarie).
        """
        n_freq_bins_stft = self.n_fft // 2 + 1 # Numero di bin di frequenza dall'STFT

        # Converti le frequenze centrali (Hz) in bin FFT (potenzialmente frazionari)
        # La formula è: bin = freq * (n_fft / sr)
        # O, se normalizzato rispetto a Nyquist: bin = (freq / (sr/2)) * (n_fft/2)
        center_bins_float = (center_freqs / (self.sr / 2.0)) * (self.n_fft / 2.0)

        # Crea i bin di frequenza dell'asse FFT come un tensore per il broadcasting [1, n_freq_bins_stft]
        fft_ax_bins = torch.arange(n_freq_bins_stft, device=device, dtype=torch.float32).unsqueeze(0)

        # Definisci i punti di sinistra (f_m_minus_1) e destra (f_m_plus_1) per ogni filtro m
        # basati sui centri dei filtri adiacenti.
        # Usa f_min e f_max (convertiti in bin) come boundaries.
        f_min_bin = (self.f_min / (self.sr / 2.0)) * (self.n_fft / 2.0)
        f_max_bin = (self.f_max / (self.sr / 2.0)) * (self.n_fft / 2.0)
        
        # Sposta su device se f_min_bin o f_max_bin non lo sono già (dovrebbero esserlo se self.f_min è buffer)
        f_min_bin = f_min_bin.to(device)
        f_max_bin = f_max_bin.to(device)

        # Punti di sinistra: [f_min_bin, center_0, center_1, ..., center_N-2]
        # Punti di destra:  [center_1, center_2, ..., center_N-1, f_max_bin]
        # center_bins_float ha shape [n_filters]
        
        left_boundaries = torch.cat((f_min_bin.unsqueeze(0), center_bins_float[:-1]))
        right_boundaries = torch.cat((center_bins_float[1:], f_max_bin.unsqueeze(0)))
        
        # Rendi le shape [n_filters, 1] per broadcasting con fft_ax_bins [1, n_freq_bins_stft]
        f_m = center_bins_float.unsqueeze(1)      # Centri f[m]
        f_m_minus_1 = left_boundaries.unsqueeze(1)  # Sinistra f[m-1]
        f_m_plus_1 = right_boundaries.unsqueeze(1)  # Destra f[m+1]

        # Calcola le pendenze (come da formula standard dei filtri triangolari/Mel)
        # H_m(k) = (k - f[m-1]) / (f[m] - f[m-1])  per la rampa ascendente
        # H_m(k) = (f[m+1] - k) / (f[m+1] - f[m])  per la rampa discendente
        # (k è fft_ax_bins)
        
        # Epsilon per stabilità numerica (evita divisione per zero se i punti coincidono)
        eps = 1e-8 

        term1_num = fft_ax_bins - f_m_minus_1
        term1_den = f_m - f_m_minus_1
        ramp_up = term1_num / (term1_den + eps)

        term2_num = f_m_plus_1 - fft_ax_bins
        term2_den = f_m_plus_1 - f_m
        ramp_down = term2_num / (term2_den + eps)
        
        # Il filtro è il minimo delle due rampe, clampato a zero.
        # Questo forma il triangolo: sale con ramp_up, scende con ramp_down.
        filter_bank = torch.max(torch.zeros_like(ramp_up), torch.min(ramp_up, ramp_down))
        
        # Normalizzazione opzionale dei filtri (es. area unitaria)
        # Le filterbank Mel di torchaudio usano 'slaney' norm che normalizza l'area a 1.
        # Per farlo in modo differenziabile:
        # filter_areas = torch.sum(filter_bank, dim=1, keepdim=True) * ( (self.sr / 2.0) / (self.n_fft / 2.0) ) # larghezza di un bin
        # filter_bank = filter_bank / (filter_areas + eps)
        # Per ora, omettiamo la normalizzazione Slaney per semplicità, dato che non era nel codice originale esplicitamente
        # per i filtri custom, ma è presente in torchaudio.functional.melscale_fbanks.

        if self.debug:
            if torch.isnan(filter_bank).any() or torch.isinf(filter_bank).any():
                logger.error("Spec: NaN/Inf in filter bank detected!")
                logger.debug("  center_freqs (Hz): %s", center_freqs.detach().cpu().numpy().round(1))
                logger.debug("  center_bins_float: %s", center_bins_float.detach().cpu().numpy().round(1))
                logger.debug("  f_m_minus_1: %s", f_m_minus_1.squeeze().detach().cpu().numpy().round(1))
                logger.debug("  f_m: %s", f_m.squeeze().detach().cpu().numpy().round(1))
                logger.debug("  f_m_plus_1: %s", f_m_plus_1.squeeze().detach().cpu().numpy().round(1))


        return filter_bank # Shape: [n_filters, n_freq_bins_stft]

    def _get_current_filter_bank(self, device):
        """
        Restituisce la filter bank corrente, usando una cache per efficienza.
        La cache viene invalidata se i parametri apprendibili cambiano.
        """
        # Controlla se i parametri sono cambiati (per invalidare la cache)
        params_have_changed = False
        if self.spec_type == "combined_log_linear":
            current_params = {
                'breakpoint': self.breakpoint.item(),
                'transition_width': self.transition_width.item()
            }
            if not self._cached_params or self._cached_params != current_params:
                params_have_changed = True
                self._cached_params = current_params

        if self._filter_bank_cache is not None and not params_have_changed:
            return self._filter_bank_cache.to(device)

        # Se non in cache o parametri cambiati, ricalcola
        if self.debug and params_have_changed:
            logger.debug("Spec: rebuilding filter bank (breakpoint=%s, transition_width=%s)", self.breakpoint.item(), self.transition_width.item())

        # Ricalcola la filter bank in base al tipo
        if self.spec_type == "mel":
            # Usa la filter bank Mel standard da torchaudio
            fb = F.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.f_min.item(),
                f_max=self.f_max.item(),
                n_mels=self.n_filters,
                sample_rate=self.sr,
                norm='slaney',
                mel_scale='htk'
            ).to(device)
            # melscale_fbanks può restituire [n_freqs, n_mels], vogliamo [n_mels, n_freqs]
            if fb.shape[0] != self.n_filters:
                fb = fb.T
            self._filter_bank_cache = fb

        elif self.spec_type == "linear_stft":
            # Per STFT lineare, non c'è una filter bank; il risultato dell'STFT viene usato direttamente.
            self._filter_bank_cache = None
            
        elif self.spec_type in ["linear_triangular", "combined_log_linear"]:
            # Entrambi questi tipi generano filtri triangolari basati su frequenze centrali.
            center_freqs = self._get_triangular_filter_centers(device=device)
            if center_freqs is None:
                raise ValueError(f"Center frequencies could not be computed for spec_type: {self.spec_type}")
            
            filter_bank = self._generate_triangular_filter_bank_from_centers(center_freqs, device=device)
            self._filter_bank_cache = filter_bank
        
        else:
            raise ValueError(f"Unsupported spec_type for filter bank generation: {self.spec_type}")

        return self._filter_bank_cache


    def forward(self, waveform):
        # waveform shape: (batch, time) o (time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # Aggiungi batch dim se manca
        
        device = waveform.device

        # 1. Calcola STFT
        # Finestra Hann è comune, assicurati sia sul device corretto
        window = torch.hann_window(self.n_fft, device=device)
        
        stft_complex = torch.stft(waveform, 
                                   n_fft=self.n_fft, 
                                   hop_length=self.hop_length,
                                   window=window,
                                   return_complex=True) # (batch, n_freq_bins, n_frames)
        
        magnitude_stft = torch.abs(stft_complex) # (batch, n_freq_bins, n_frames)

        if self.spec_type == "linear_stft":
            # Per STFT lineare, usiamo direttamente la magnitudine (o la sua versione in dB)
            # Non applichiamo un'ulteriore filter bank qui.
            # La conversione in dB è spesso fatta dopo, nel modello principale.
            spectrogram = magnitude_stft # Shape: (batch, n_fft // 2 + 1, n_frames)
        else:
            # 2. Genera/Ottieni la filter bank corrente
            # filter_bank ha shape [n_filters, n_freq_bins_stft]
            filter_bank = self._get_current_filter_bank(device=device) 

            if filter_bank is None:
                raise ValueError(f"Filter bank could not be created for spec_type {self.spec_type}")

            # 3. Applica la filter bank
            # magnitude_stft: (batch, n_freq_bins, n_frames)
            # filter_bank: (n_filters, n_freq_bins)
            # Risultato desiderato: (batch, n_filters, n_frames)
            # torch.matmul gestisce il broadcasting: (F,K) @ (B,K,T) -> (B,F,T)
            spectrogram = torch.matmul(filter_bank, magnitude_stft)
        
        # La conversione in dB (AmplitudeToDB) è tipicamente un layer separato nel modello principale.
        # Se volessimo farla qui:
        # spectrogram = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)(spectrogram)
        
        return spectrogram


# --- Funzioni di utilità e test (possono rimanere per debug) ---

# Esempio di come caricare audio (se necessario per test standalone)
def load_audio_waveform(file_path, target_sr):
    waveform, sr = torchaudio.load(file_path)
    if waveform.dim() > 1 and waveform.shape[0] > 1: # Converti a mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr: # Resample se necessario
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze()


def test_differentiable_spectrogram_module():
    logger.info("Testing DifferentiableSpectrogram module...")
    sr = 24000
    duration = 2 # secondi
    n_samples = sr * duration
    
    # Test combined_log_linear (trainable)
    logger.info("\n--- Test 1: combined_log_linear (trainable) ---")
    spec_module_trainable = DifferentiableSpectrogram(
        sr=sr, n_filters=40, f_min=100.0, f_max=10000.0,
        n_fft=1024, hop_length=240,
        initial_breakpoint=3000.0, initial_transition_width=200.0,
        trainable_filterbank=True, spec_type="combined_log_linear", debug=True
    ).eval() # Metti in eval mode per testare forward

    # Verifica parametri
    if hasattr(spec_module_trainable, 'breakpoint') and isinstance(spec_module_trainable.breakpoint, nn.Parameter):
        logger.debug("  Breakpoint: %s (is nn.Parameter)", spec_module_trainable.breakpoint.item())
        spec_module_trainable.breakpoint.data.fill_(2000.0) # Modifica per test
        logger.debug("  Breakpoint modified to: %s", spec_module_trainable.breakpoint.item())

    if hasattr(spec_module_trainable, 'transition_width') and isinstance(spec_module_trainable.transition_width, nn.Parameter):
        logger.debug("  Transition Width: %s (is nn.Parameter)", spec_module_trainable.transition_width.item())
        spec_module_trainable.transition_width.data.fill_(50.0)
        logger.debug("  Transition Width modified to: %s", spec_module_trainable.transition_width.item())

    waveform_test = torch.randn(n_samples) * 0.1 # Batch di 1, o singolo waveform
    
    # Test forward pass
    try:
        spec_output = spec_module_trainable(waveform_test.unsqueeze(0)) # Aggiungi batch dim
        logger.info("  Output spectrogram shape: %s", tuple(spec_output.shape)) # Expected: [1, n_filters, n_frames]
        assert spec_output.shape[1] == spec_module_trainable.n_filters
        # Tentativo di backward pass
        if spec_module_trainable.trainable_filterbank:
            spec_output.sum().backward()
            logger.info("  Backward pass successful.")
            if spec_module_trainable.breakpoint.grad is not None:
                logger.debug("    Gradient for breakpoint: %s", spec_module_trainable.breakpoint.grad.item())
            else:
                logger.warning("    No gradient for breakpoint.")
            if spec_module_trainable.transition_width.grad is not None:
                logger.debug("    Gradient for transition_width: %s", spec_module_trainable.transition_width.grad.item())
            else:
                logger.warning("    No gradient for transition_width.")
        else:
            logger.info("  Trainable filterbank is False, skipping gradient check.")

    except Exception as e:
        logger.error("  Error during trainable combined_log_linear test: %s", e)
        import traceback
        traceback.print_exc()

    # Test Mel (non trainable breakpoint/TW)
    logger.info("\n--- Test 2: mel (non-trainable breakpoint/TW) ---")
    spec_module_mel = DifferentiableSpectrogram(
        sr=sr, n_filters=64, f_min=50.0, f_max=12000.0,
        n_fft=1024, hop_length=240,
        spec_type="mel", trainable_filterbank=False, debug=True # trainable_filterbank non ha effetto su Mel qui
    ).eval()
    try:
        spec_output_mel = spec_module_mel(waveform_test.unsqueeze(0))
        logger.info("  Mel output shape: %s", tuple(spec_output_mel.shape))
        assert spec_output_mel.shape[1] == spec_module_mel.n_filters
    except Exception as e:
        logger.error("  Error during mel test: %s", e)

    # Test linear_triangular (non trainable breakpoint/TW)
    logger.info("\n--- Test 3: linear_triangular (non-trainable breakpoint/TW) ---")
    spec_module_lin = DifferentiableSpectrogram(
        sr=sr, n_filters=50, f_min=0.0, f_max=16000.0, # f_min=0 per lineare completo
        n_fft=1024, hop_length=240,
        spec_type="linear_triangular", trainable_filterbank=False, debug=True
    ).eval()
    try:
        spec_output_lin = spec_module_lin(waveform_test.unsqueeze(0))
        logger.info("  Linear triangular output shape: %s", tuple(spec_output_lin.shape))
        assert spec_output_lin.shape[1] == spec_module_lin.n_filters
    except Exception as e:
        logger.error("  Error during linear triangular test: %s", e)

    # Test linear_stft (no filterbank applied in this module)
    logger.info("\n--- Test 4: linear_stft ---")
    spec_module_stft = DifferentiableSpectrogram(
        sr=sr, n_fft=1024, hop_length=240,
        spec_type="linear_stft" 
        # n_filters, f_min, f_max non sono usati per linear_stft qui
    ).eval()
    try:
        spec_output_stft = spec_module_stft(waveform_test.unsqueeze(0))
        expected_freq_bins = spec_module_stft.n_fft // 2 + 1
        logger.info("  Linear STFT output shape: %s", tuple(spec_output_stft.shape)) # Expected [1, n_fft/2+1, n_frames]
        assert spec_output_stft.shape[1] == expected_freq_bins
    except Exception as e:
        logger.error("  Error during linear STFT test: %s", e)


def create_spectrogram_module(config):
    """Factory function per creare il modulo spectrogram corretto"""
    
    spec_type = config.get('spectrogram_type', 'combined_log_linear')
    
    if spec_type == 'combined_log_linear':
        # Usa la classe esistente
        return DifferentiableSpectrogram(
            sr=config.get('sample_rate', 24000),
            n_filters=config.get('n_linear_filters', 64),
            f_min=config.get('f_min', 150.0),
            f_max=config.get('f_max', 10000.0),
            n_fft=config.get('n_fft', 1024),
            hop_length=config.get('hop_length', 240),
            initial_breakpoint=config.get('initial_breakpoint', 4000.0),
            initial_transition_width=config.get('initial_transition_width', 100.0),
            trainable_filterbank=config.get('trainable_filterbank', True),
            spec_type=spec_type,
            debug=config.get('debug', False)
        )
    
    elif spec_type == 'fully_learnable':
        # Usa la nuova classe
        n_filters = config.get('n_linear_filters', 64)
        n_fft = config.get('n_fft', 1024)
        n_freq_bins = n_fft // 2 + 1
        sample_rate = config.get('sample_rate', 24000)
        hop_length = config.get('hop_length', 240)
        init_strategy = config.get('filter_init_strategy', 'triangular_noise')
        
        return FullyLearnableFilterBank(
            n_filters=n_filters,
            n_freq_bins=n_freq_bins,
            sample_rate=sample_rate,
            init_strategy=init_strategy,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    elif spec_type in ['mel', 'linear_triangular', 'linear_stft']:
        # Altri tipi esistenti
        return DifferentiableSpectrogram(
            sr=config.get('sample_rate', 24000),
            n_filters=config.get('n_linear_filters', 64),
            f_min=config.get('f_min', 150.0),
            f_max=config.get('f_max', 10000.0),
            n_fft=config.get('n_fft', 1024),
            hop_length=config.get('hop_length', 240),
            trainable_filterbank=config.get('trainable_filterbank', True),
            spec_type=spec_type,
            debug=config.get('debug', False)
        )
    
    else:
        raise ValueError(f"Unknown spectrogram type: {spec_type}")


if __name__ == '__main__':
    # Esegui i test se lo script è chiamato direttamente
    test_differentiable_spectrogram_module()

    # Esempio d'uso con un file audio reale (opzionale)
    # file_path = "your_audio_file.wav" # Sostituisci con un tuo file
    # target_sr_test = 16000
    # try:
    #     waveform = load_audio_waveform(file_path, target_sr_test)
    #     spec_module_real = DifferentiableSpectrogram(sr=target_sr_test, spec_type="combined_log_linear", trainable_filterbank=True).eval()
    #     spec_real = spec_module_real(waveform.unsqueeze(0))
    #     print(f"\nSpectrogram from real audio ({file_path}), shape: {spec_real.shape}")

    #     # Visualizzazione opzionale (richiede matplotlib)
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 4))
    #     plt.imshow(torchaudio.transforms.AmplitudeToDB()(spec_real.squeeze().cpu().detach()).numpy(), 
    #                aspect='auto', origin='lower', cmap='viridis')
    #     plt.title("Differentiable Spectrogram from Real Audio")
    #     plt.xlabel("Frames")
    #     plt.ylabel("Filter Index")
    #     plt.colorbar(label="Magnitude (dB)")
    #     plt.tight_layout()
    #     plt.show()
    # except FileNotFoundError:
    #     print(f"\nFile {file_path} non trovato. Salto test con audio reale.")
    # except Exception as e:
    #     print(f"\nErrore durante test con audio reale: {e}")
