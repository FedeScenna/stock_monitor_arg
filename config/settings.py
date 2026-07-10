from pathlib import Path

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR         = PROJECT_ROOT / "data" / "cedears"
PORTFOLIO_DIR    = PROJECT_ROOT / "data" / "portfolio"
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "fundamentals"

START_DATE = "2000-01-01"

# ---------------------------------------------------------------------------
# Weekly screen parameters (scripts/weekly_screen.py + app.py "Weekly Screen")
# ---------------------------------------------------------------------------
P_TBV_MAX      = 1.0   # "priced at tangible value": price <= tangible book / share
RSI_BUY        = 30    # RSI <= this -> oversold / buy zone
RSI_SELL       = 70    # RSI >= this -> overbought / sell zone
WMA_WEEKS      = 20    # weekly moving-average length (20-week MA)
EMA_FAST       = 9     # fast "dynamic mean" (EMA9)
EMA_SLOW       = 21    # slow "dynamic mean" (EMA21)
CROSS_LOOKBACK = 5     # a cross counts if it happened within the last N trading days
MIN_DOLLAR_VOL = 0     # liquidity floor: drop names below this 20-day avg $ volume (0 = off)
SCREEN_MIN_ROWS = 50   # minimum OHLCV rows required to screen a ticker

# ---------------------------------------------------------------------------
# Murphy technical-analysis strategies (src/screening/murphy.py,
# scripts/technical_signals.py + app.py "Technical Signals").
# Indicator toolkit from John J. Murphy, "Technical Analysis of the Financial
# Markets": directional movement (ADX/DMI), stochastics, Williams %R, ROC,
# On-Balance Volume, ATR, Donchian breakout.
# ---------------------------------------------------------------------------
ATR_LEN            = 14    # Wilder Average True Range length
ADX_LEN            = 14    # Wilder directional-movement / ADX length
ADX_TREND_MIN      = 25    # ADX >= this => trending market (DI cross is tradable)
STOCH_K            = 14    # stochastic %K lookback
STOCH_D            = 3     # stochastic %D smoothing
STOCH_SMOOTH       = 3     # slow-%K smoothing
STOCH_OVERSOLD     = 20    # stochastic oversold threshold
STOCH_OVERBOUGHT   = 80    # stochastic overbought threshold
WILLIAMS_LEN       = 14    # Williams %R lookback
WILLIAMS_OVERSOLD  = -80   # %R <= this => oversold
WILLIAMS_OVERBOUGHT = -20  # %R >= this => overbought
ROC_LEN            = 12    # rate-of-change / momentum lookback
OBV_SLOPE_LEN      = 20    # OBV slope window (confirm/deny the price trend)
DONCHIAN_LEN       = 252   # Donchian / 52-week breakout channel (~1 year)

# ---------------------------------------------------------------------------
# Multi-model forecasting (src/forecasting/, scripts/forecast_benchmark.py,
# scripts/ensemble_forecast.py). Deep time-series models (Nixtla N-HiTS /
# PatchTST / TFT) are benchmarked against Kronos, then blended into an ensemble.
# ---------------------------------------------------------------------------
FORECAST_HORIZON   = 21    # trading days to forecast (~1 month), matches Kronos
FORECAST_INPUT     = 252   # context window (lookback) fed to the neural models
FORECAST_MAX_STEPS = 200   # training steps per neural model fit (speed vs accuracy)
FORECAST_QUANTILES = [0.1, 0.5, 0.9]   # probabilistic band (q50 = point forecast)
NEURAL_MODELS      = ["NHITS", "PatchTST", "TFT"]   # Nixtla models in the ensemble
ENSEMBLE_MIN_ROWS  = 400   # minimum OHLCV rows to train the neural forecasters

# ---------------------------------------------------------------------------
# Complete CEDEAR universe traded on BYMA
# Source: comafi.com.ar/custodiaglobal/Programas-CEDEARs-2483.note.aspx
# Keys are yfinance-compatible tickers (may differ from BYMA ticker).
# See BYMA_TICKER_MAP below for the BYMA↔yfinance mapping.
# ---------------------------------------------------------------------------
CEDEAR_TICKERS: dict[str, str] = {
    # A
    "MMM":   "3M Co",
    "ABT":   "Abbott Laboratories",
    "ABBV":  "AbbVie Inc",
    "ANF":   "Abercrombie & Fitch",
    "ACN":   "Accenture PLC",
    "AGRO":  "Adecoagro SA",          # BYMA: ADGO
    "ADDYY": "Adidas AG",             # BYMA: ADS  (US OTC ADR)
    "ADBE":  "Adobe Inc",
    "AAP":   "Advance Auto Parts",
    "AMD":   "Advanced Micro Devices",
    "AEG":   "Aegon Ltd",
    "AEM":   "Agnico Eagle Mines",
    "BABA":  "Alibaba Group",
    "GOOGL": "Alphabet Inc",
    "MO":    "Altria Group",
    "AMZN":  "Amazon.com Inc",
    "ABEV":  "Ambev SA",
    "AMX":   "America Movil",
    "AAL":   "American Airlines",
    "AXP":   "American Express",
    "AIG":   "American International Group",
    "AMGN":  "Amgen Inc",
    "ADI":   "Analog Devices",
    "AAPL":  "Apple Inc",
    "AMAT":  "Applied Materials",
    "ARCO":  "Arcos Dorados Holdings",
    "ARM":   "ARM Holdings PLC",
    "AZN":   "AstraZeneca PLC",
    "T":     "AT&T Inc",
    "ADP":   "Automatic Data Processing",
    "AVY":   "Avery Dennison Corp",
    "CAR":   "Avis Budget Group",
    # B
    "BIDU":  "Baidu Inc",
    "BKR":   "Baker Hughes",
    "BBVA":  "Banco Bilbao Vizcaya",
    "BBD":   "Banco Bradesco",
    "BSBR":  "Banco Santander Brasil",
    "SAN":   "Banco Santander SA",
    "BAC":   "Bank of America",
    "BK":    "Bank of New York Mellon",
    "BCS":   "Barclays PLC",
    "GOLD":  "Barrick Gold Corp",
    "BASFY": "BASF SE",               # BYMA: BAS  (US OTC ADR)
    "BAYRY": "Bayer AG",              # BYMA: BAYN (US OTC ADR)
    "BRK-B": "Berkshire Hathaway B",  # BYMA: BRKB
    "BHP":   "BHP Group",
    "BIOX":  "Bioceres Crop Solutions",
    "BIIB":  "Biogen Inc",
    "BB":    "BlackBerry Ltd",
    "XYZ":   "Block Inc",             # BYMA: XYZ  (formerly SQ)
    "BA":    "Boeing",
    "BKNG":  "Booking Holdings",
    "BP":    "BP PLC",
    "LND":   "BrasilAgro",
    "BAK":   "Braskem SA",
    "BRFS":  "BRF SA",
    "BMY":   "Bristol-Myers Squibb",
    "AVGO":  "Broadcom Inc",
    "BNG":   "Bunge Global SA",
    # C
    "CAH":   "Cardinal Health",
    "CCL":   "Carnival Corp",
    "CAT":   "Caterpillar Inc",
    "CX":    "Cemex SAB",
    "EBR":   "Centrais Eletricas Brasileiras",
    "SCHW":  "Charles Schwab",
    "CVX":   "Chevron Corp",
    "CBD":   "Cia Brasileira de Distribuicao",
    "SBS":   "Cia Saneamento Basico",
    "SID":   "Cia Siderurgica Nacional",
    "CSCO":  "Cisco Systems",
    "C":     "Citigroup Inc",
    "KO":    "Coca-Cola Co",
    "KOF":   "Coca-Cola FEMSA",       # BYMA: KOFM
    "CDE":   "Coeur Mining",
    "COIN":  "Coinbase Global",
    "CL":    "Colgate-Palmolive",
    "XLC":   "Communication Services Select Sector ETF",
    "ELP":   "Companhia Paranaense de Energia",
    "XLY":   "Consumer Discretionary Select Sector ETF",
    "XLP":   "Consumer Staples Select Sector ETF",
    "GLW":   "Corning Inc",
    "CAAP":  "Corp America Airports",
    "COST":  "Costco Wholesale",
    "CVS":   "CVS Health",
    # D
    "DHR":   "Danaher Corp",
    "DANOY": "Danone",                # BYMA: BSN  (US OTC ADR)
    "DE":    "Deere & Co",
    "DAL":   "Delta Air Lines",
    "DESP":  "Despegar.com",
    "DTEGY": "Deutsche Telekom AG",   # BYMA: DTEA (US OTC ADR)
    "DEO":   "Diageo PLC",
    "DOCU":  "DocuSign Inc",
    "DOW":   "Dow Inc",
    "DD":    "DuPont de Nemours",
    # E
    "EONGY": "E.ON SE",               # BYMA: EOAN (US OTC ADR)
    "EBAY":  "eBay Inc",
    "EA":    "Electronic Arts",
    "LLY":   "Eli Lilly & Co",
    "ERJ":   "Embraer SA",
    "E":     "ENI SPA",
    "EFX":   "Equifax Inc",
    "EQNR":  "Equinor ASA",
    "ERIC":  "Ericsson LM",
    "ETSY":  "Etsy Inc",
    "XOM":   "Exxon Mobil",
    # F
    "FNMA":  "Fannie Mae",
    "FDX":   "FedEx Corp",
    "RACE":  "Ferrari NV",
    "FSLR":  "First Solar Inc",
    "FMX":   "Fomento Economico Mexicano",
    "FMCC":  "Freddie Mac",
    "FCX":   "Freeport-McMoRan",
    # G
    "GRMN":  "Garmin Ltd",
    "GE":    "GE Aerospace",
    "GM":    "General Motors",
    "GPRK":  "GeoPark Ltd",
    "GGB":   "Gerdau SA",
    "GILD":  "Gilead Sciences",
    "GLOB":  "Globant SA",
    "GFI":   "Gold Fields Ltd",
    "GS":    "Goldman Sachs",
    "GT":    "Goodyear Tire & Rubber",
    "PAC":   "Grupo Aeroportuario Pacifico",
    "ASR":   "Grupo Aeroportuario Sur",
    "TV":    "Grupo Televisa",
    "GSK":   "GSK PLC",
    # H
    "HAL":   "Halliburton Co",
    "HOG":   "Harley-Davidson Inc",
    "HMY":   "Harmony Gold Mining",
    "HDB":   "HDFC Bank Ltd",
    "XLV":   "Health Care Select Sector ETF",
    "HL":    "Hecla Mining",
    "HSY":   "Hershey Co",
    "HD":    "Home Depot",
    "HMC":   "Honda Motor Co",
    "HON":   "Honeywell International",
    "HWM":   "Howmet Aerospace",
    "HPQ":   "HP Inc",
    "HSBC":  "HSBC Holdings",
    # I
    "IBN":   "ICICI Bank Ltd",
    "XLI":   "Industrial Select Sector ETF",
    "INFY":  "Infosys Ltd",
    "ING":   "ING Groep",
    "INTC":  "Intel Corp",
    "IP":    "International Paper",
    "IBM":   "IBM",
    "IFF":   "Intl Flavors & Fragrances",
    "ISRG":  "Intuitive Surgical",
    "IBIT":  "iShares Bitcoin Trust",
    "FXI":   "iShares China Large-Cap ETF",
    "NOW":   "ServiceNow Inc",
    "RSP":   "Invesco S&P 500 Equal Weight ETF",
    "SPY":   "SPDR S&P 500 ETF Trust",
    "IEUR":  "iShares Core MSCI Europe ETF",
    "IBB":   "iShares NASDAQ Biotechnology ETF",
    "IVW":   "iShares S&P 500 Growth ETF",
    "IVE":   "iShares S&P 500 Value ETF",
    "ITUB":  "Itau Unibanco",
    # J
    "JD":    "JD.com Inc",
    "JNJ":   "Johnson & Johnson",
    "JCI":   "Johnson Controls",
    "YY":    "JOYY Inc",
    "JPM":   "JPMorgan Chase",
    # K
    "KB":    "KB Financial Group",
    "KMB":   "Kimberly-Clark",
    "KGC":   "Kinross Gold",
    "PHG":   "Koninklijke Philips",
    "KEP":   "Korea Electric Power",
    # L
    "LRCX":  "Lam Research",
    "LVS":   "Las Vegas Sands",
    "LAC":   "Lithium Americas Corp",
    "LAAC":  "Lithium Argentina AG",  # BYMA: LAR
    "LYG":   "Lloyds Banking Group",
    "LMT":   "Lockheed Martin",
    # M
    "MMC":   "Marsh & McLennan",
    "MRVL":  "Marvell Technology",
    "XLB":   "Materials Select Sector ETF",
    "MA":    "Mastercard Inc",
    "MCD":   "McDonald's Corp",
    "MUX":   "McEwen Mining",
    "MDT":   "Medtronic PLC",
    "MELI":  "MercadoLibre Inc",
    "MBGAF": "Mercedes-Benz Group AG", # BYMA: MBG (US OTC ADR)
    "MRK":   "Merck & Co",
    "META":  "Meta Platforms",
    "MSFT":  "Microsoft Corp",
    "MUFG":  "Mitsubishi UFJ Financial",
    "MFG":   "Mizuho Financial Group",
    "MRNA":  "Moderna Inc",
    "MDLZ":  "Mondelez International",
    "MSI":   "Motorola Solutions",
    # N
    "NGG":   "National Grid PLC",
    "NTES":  "NetEase Inc",
    "NFLX":  "Netflix Inc",
    "NEM":   "Newmont Corp",
    "NXE":   "NexGen Energy",
    "NKE":   "Nike Inc",
    "NIO":   "NIO Inc",
    "NSANY": "Nissan Motor Co",        # BYMA: NSAN (US OTC ADR)
    "NOK":   "Nokia Corp",             # BYMA: NOKA
    "NMR":   "Nomura Holdings",
    "NG":    "NovaGold Resources",
    "NVS":   "Novartis AG",
    "NU":    "Nu Holdings Ltd",
    "NUE":   "Nucor Corp",
    "NVDA":  "Nvidia Corp",
    # O
    "ORLY":  "O'Reilly Automotive",
    "ORCL":  "Oracle Corp",
    "ORAN":  "Orange SA",              # BYMA: ORANY
    # P
    "PCAR":  "PACCAR Inc",
    "PAGS":  "PagSeguro Digital",
    "PLTR":  "Palantir Technologies",
    "PAAS":  "Pan American Silver",
    "PYPL":  "PayPal Holdings",
    "PSO":   "Pearson PLC",
    "PEP":   "PepsiCo Inc",
    "PBR":   "Petrobras",
    "PFE":   "Pfizer Inc",
    "PM":    "Philip Morris International",
    "PSX":   "Phillips 66",
    "PINS":  "Pinterest Inc",
    "PBI":   "Pitney Bowes",
    "PKX":   "POSCO Holdings",         # BYMA: PKS
    "PG":    "Procter & Gamble",
    # Q
    "QCOM":  "Qualcomm Inc",
    # R
    "XLRE":  "Real Estate Select Sector ETF",
    "RIO":   "Rio Tinto PLC",
    "RIOT":  "Riot Platforms",
    "ROKU":  "Roku Inc",
    "ROST":  "Ross Stores",
    "RTX":   "RTX Corporation",
    # S
    "CRM":   "Salesforce Inc",
    "SAP":   "SAP SE",
    "SLB":   "Schlumberger Ltd",
    "SE":    "Sea Ltd",
    "SHEL":  "Shell PLC",
    "SHOP":  "Shopify Inc",
    "SIEGY": "Siemens AG",
    "SWKS":  "Skyworks Solutions",
    "SNAP":  "Snap Inc",
    "SNA":   "Snap-on Inc",
    "SNOW":  "Snowflake Inc",
    "SONY":  "Sony Corp",
    "SCCO":  "Southern Copper Corp",
    "SPOT":  "Spotify Technology",
    "SPGI":  "S&P Global Inc",
    "SBUX":  "Starbucks Corp",
    "STLA":  "Stellantis NV",
    "STNE":  "StoneCo Ltd",
    "SUZ":   "Suzano SA",
    "SYY":   "Sysco Corp",
    # T
    "TMUS":  "T-Mobile US",
    "TSM":   "Taiwan Semiconductor",
    "TGT":   "Target Corp",
    "XLK":   "Technology Select Sector ETF",
    "TEF":   "Telefonica SA",          # BYMA: TEFO
    "VIV":   "Telefonica Brasil",
    "TS":    "Tenaris SA",             # BYMA: TEN
    "TX":    "Ternium SA",             # BYMA: TXR
    "TSLA":  "Tesla Inc",
    "TXN":   "Texas Instruments",
    "TMO":   "Thermo Fisher Scientific",
    "TSU":   "TIM SA",
    "TJX":   "TJX Companies",
    "TTE":   "TotalEnergies SE",
    "TM":    "Toyota Motor Corp",
    "TRV":   "Travelers Companies",
    "TCOM":  "Trip.com Group",
    "TRIP":  "Tripadvisor Inc",
    "TWLO":  "Twilio Inc",
    # U
    "UGP":   "Ultrapar Participacoes",
    "UL":    "Unilever NV",
    "UNP":   "Union Pacific Corp",
    "X":     "US Steel Corp",
    "UNH":   "UnitedHealth Group",
    "URBN":  "Urban Outfitters",
    "USB":   "US Bancorp",
    # V
    "VALE":  "Vale SA",
    "VEA":   "Vanguard FTSE Developed Markets ETF",
    "VRSN":  "VeriSign Inc",
    "VZ":    "Verizon Communications",
    "SPCE":  "Virgin Galactic Holdings",
    "V":     "Visa Inc",
    "VIST":  "Vista Energy",
    "VOD":   "Vodafone Group",
    # W
    "WBA":   "Walgreens Boots Alliance",
    "WMT":   "Walmart Inc",
    "DIS":   "Walt Disney Co",         # BYMA: DISN
    "WB":    "Weibo Corp",             # BYMA: WBO
    "WFC":   "Wells Fargo",
    # X
    "XRX":   "Xerox Holdings",
    "XP":    "XP Inc",
    # Y
    "YELP":  "Yelp Inc",
    # Z
    "ZM":    "Zoom Video Communications",
}

# Tickers to skip during data downloads (Russian sanctioned/delisted,
# bankrupt, or no viable yfinance mapping).
CEDEAR_SKIP: set[str] = {
    "OGZD",   # Gazprom — sanctioned/delisted
    "LKOD",   # Lukoil — sanctioned/delisted
    "MBT",    # MTS — sanctioned/delisted
    "NLM",    # Novolipetsk Steel — sanctioned/delisted
    "ATAD",   # Tatneft — sanctioned/delisted
    "SHPWQ",  # Shapeways — bankrupt
    "HHPD",   # Hon Hai GDR — no yfinance mapping
    "SMSN",   # Samsung GDR — no yfinance mapping
    "NEC1",   # NEC Corp GDR — no yfinance mapping
    "AKO.B",  # Embotelladora Andina — dot in ticker unsupported
    "YZCA",   # Yanzhou Coal — no yfinance mapping
}

# BYMA ticker → yfinance ticker (for reference/display only).
# Data files use the yfinance key from CEDEAR_TICKERS above.
BYMA_TO_YFINANCE: dict[str, str] = {
    "ADGO":  "AGRO",
    "ADS":   "ADDYY",
    "BAS":   "BASFY",
    "BAYN":  "BAYRY",
    "BRKB":  "BRK-B",
    "BSN":   "DANOY",
    "DISN":  "DIS",
    "DTEA":  "DTEGY",
    "EOAN":  "EONGY",
    "KOFM":  "KOF",
    "LAR":   "LAAC",
    "MBG":   "MBGAF",
    "NOKA":  "NOK",
    "NSAN":  "NSANY",
    "ORANY": "ORAN",
    "PKS":   "PKX",
    "TEFO":  "TEF",
    "TEN":   "TS",
    "TXR":   "TX",
    "WBO":   "WB",
    "XYZ":   "XYZ",   # same — Block Inc rebranded to XYZ on NYSE
}

# ---------------------------------------------------------------------------
# Portfolio holdings (subset of CEDEAR_TICKERS + local Argentine ADRs)
# ---------------------------------------------------------------------------
PORTFOLIO_CEDEARS: dict[str, str] = {
    "AAPL":  "Apple Inc",
    "AMZN":  "Amazon.com Inc",
    "AVGO":  "Broadcom Inc",
    "FXI":   "iShares China Large-Cap ETF",
    "GOOGL": "Alphabet Inc",
    "MELI":  "MercadoLibre Inc",
    "MSFT":  "Microsoft Corp",
    "NOW":   "ServiceNow Inc",
    "NU":    "Nu Holdings Ltd",
    "NVDA":  "Nvidia Corp",
    "RSP":   "Invesco S&P 500 Equal Weight ETF",
    "SPY":   "SPDR S&P 500 ETF Trust",
    "TSM":   "Taiwan Semiconductor",
    "VIST":  "Vista Energy ADS",
    "XLE":   "Energy Select Sector SPDR",
}

# Argentine local stocks tracked alongside CEDEARs
PORTFOLIO_STOCKS: dict[str, str] = {
    "PAM": "Pampa Energia (PAMP ADR)",
    "YPF": "YPF SA (YPFD ADR)",
}

# Combined portfolio assets
PORTFOLIO_ASSETS: dict[str, str] = {**PORTFOLIO_CEDEARS, **PORTFOLIO_STOCKS}

# All tickers (CEDEAR universe + portfolio stocks) — used by app.py / fetch scripts
TICKERS: dict[str, str] = {**CEDEAR_TICKERS, **PORTFOLIO_STOCKS}
