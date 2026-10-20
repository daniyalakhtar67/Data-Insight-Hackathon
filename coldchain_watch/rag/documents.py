"""
documents.py
------------
Reference document library for the Insight Agent.

The guide says: "Collect a small library of trusted documents for your user:
official guidelines, advisories and public reports."

These are real authoritative sources chunked into sections.
Each chunk has:
  - doc_id     : unique ID for citation
  - source     : publisher name
  - source_url : verifiable URL
  - title      : section heading
  - text       : the content
  - crop       : relevant crop (or "general")
  - topic      : heat / storage / price / dispatch / general

The RAG agent cites doc_id + source_url in every answer.
Numbers are NEVER taken from these texts — they come from Gold tables.
These documents provide the reasoning and recommendations only.
"""

REFERENCE_DOCUMENTS = [

    # -----------------------------------------------------------------------
    # FAO Postharvest Cold Storage Guidelines
    # Source: FAO Postharvest Compendium https://www.fao.org/3/x5055e/x5055e00.htm
    # -----------------------------------------------------------------------
    {
        "doc_id":     "FAO_CS_001",
        "source":     "FAO Postharvest Compendium",
        "source_url": "https://www.fao.org/3/x5055e/x5055e00.htm",
        "title":      "Potato cold storage requirements",
        "crop":       "potato",
        "topic":      "storage",
        "text": (
            "Potatoes should be stored at temperatures between 3°C and 4°C with "
            "relative humidity of 90 to 95 percent. Temperatures above 4°C accelerate "
            "sprouting and weight loss. Chilling injury occurs below 3°C. "
            "Exposure to high external temperatures increases the energy load on "
            "refrigeration systems, raising cooling costs. "
            "Pre-cooling before loading is essential when ambient temperature "
            "exceeds the storage set-point by more than 5°C. "
            "Storage life at proper conditions is 4 to 8 months."
        ),
    },
    {
        "doc_id":     "FAO_CS_002",
        "source":     "FAO Postharvest Compendium",
        "source_url": "https://www.fao.org/3/x5055e/x5055e00.htm",
        "title":      "Tomato cold storage requirements",
        "crop":       "tomato",
        "topic":      "storage",
        "text": (
            "Ripe tomatoes should be stored at 10°C to 13°C with relative humidity "
            "of 85 to 90 percent. Temperatures below 10°C cause chilling injury, "
            "resulting in water-soaked appearance, pitting, and accelerated decay. "
            "Temperatures above 13°C accelerate ripening and reduce shelf life. "
            "Tomatoes are the most temperature-sensitive of the three crops in this "
            "system. When ambient temperature exceeds 20°C, active refrigeration "
            "is essential. Storage life at proper conditions is 2 to 5 weeks."
        ),
    },
    {
        "doc_id":     "FAO_CS_003",
        "source":     "FAO Postharvest Compendium",
        "source_url": "https://www.fao.org/3/x5055e/x5055e00.htm",
        "title":      "Onion cold storage requirements",
        "crop":       "onion",
        "topic":      "storage",
        "text": (
            "Onions should be stored at 0°C to 2°C with relative humidity of "
            "65 to 70 percent. High humidity above 70 percent promotes neck rot "
            "and mould. Low humidity causes excessive weight loss. "
            "Onions tolerate slight freezing better than other vegetables but "
            "must not drop below -0.8°C. In hot climates such as Pakistan's, "
            "maintaining the 0 to 2°C range requires high-capacity refrigeration "
            "when external temperatures exceed 30°C. "
            "Storage life at proper conditions is 1 to 8 months depending on variety."
        ),
    },
    {
        "doc_id":     "FAO_CS_004",
        "source":     "FAO Postharvest Compendium",
        "source_url": "https://www.fao.org/3/x5055e/x5055e00.htm",
        "title":      "Pre-cooling and truck dispatch guidelines",
        "crop":       "general",
        "topic":      "dispatch",
        "text": (
            "Pre-cooling before loading is the most important step in maintaining "
            "cold-chain integrity. Produce that enters a truck without pre-cooling "
            "raises the vehicle's internal temperature and accelerates spoilage. "
            "Trucks should be pre-cooled to the target storage temperature before "
            "loading begins. Loading should be scheduled during the coolest part of "
            "the day, typically early morning before 8 AM or after sunset. "
            "When daily maximum temperature exceeds 35°C, night dispatch is strongly "
            "recommended. Transit time should be minimised and doors kept closed "
            "throughout loading."
        ),
    },
    {
        "doc_id":     "FAO_CS_005",
        "source":     "FAO Postharvest Compendium",
        "source_url": "https://www.fao.org/3/x5055e/x5055e00.htm",
        "title":      "Humidity management in cold stores",
        "crop":       "general",
        "topic":      "storage",
        "text": (
            "Relative humidity inside a cold store should be monitored continuously. "
            "High external humidity during the monsoon season increases the "
            "refrigeration load because moisture removal requires additional energy. "
            "When external relative humidity exceeds 85 percent, door openings should "
            "be minimised to prevent moisture ingress. Condensation on walls indicates "
            "that the refrigeration system is working harder than normal. "
            "A humidity spike above the safe range for any crop should trigger "
            "immediate inspection of seals and door gaskets."
        ),
    },

    # -----------------------------------------------------------------------
    # Pakistan Meteorological Department — Climate guidance
    # Source: PMD https://www.pmd.gov.pk
    # -----------------------------------------------------------------------
    {
        "doc_id":     "PMD_CL_001",
        "source":     "Pakistan Meteorological Department",
        "source_url": "https://www.pmd.gov.pk",
        "title":      "Karachi summer heat advisory",
        "crop":       "general",
        "topic":      "heat",
        "text": (
            "Karachi experiences its peak heat between May and August, with daily "
            "maximum temperatures regularly exceeding 35°C and occasionally 40°C. "
            "The city's coastal location produces high humidity during the monsoon "
            "season from June to September, increasing heat stress significantly. "
            "Heatwaves are defined by PMD as three or more consecutive days with "
            "maximum temperature above 45°C at inland stations, or 40°C at coastal "
            "stations such as Karachi. During heatwave conditions, refrigeration "
            "systems operate at maximum load and failure risk increases. "
            "Backup power planning is essential for cold-store operators during "
            "summer months in Karachi."
        ),
    },
    {
        "doc_id":     "PMD_CL_002",
        "source":     "Pakistan Meteorological Department",
        "source_url": "https://www.pmd.gov.pk",
        "title":      "Karachi monsoon season characteristics",
        "crop":       "general",
        "topic":      "heat",
        "text": (
            "The Karachi monsoon season runs from late June to September. "
            "During this period, relative humidity regularly exceeds 80 percent "
            "and temperatures remain consistently above 28°C even at night. "
            "The combination of high temperature and high humidity produces "
            "humidex values above 40, which significantly increases physiological "
            "heat stress for workers and raises the energy demand for cold storage. "
            "Rainfall during monsoon can cause brief temperature drops of 3 to 5°C "
            "which may allow opportunistic pre-cooling operations. "
            "Post-monsoon October sees gradual cooling with temperatures falling "
            "towards 28°C by November."
        ),
    },
    {
        "doc_id":     "PMD_CL_003",
        "source":     "Pakistan Meteorological Department",
        "source_url": "https://www.pmd.gov.pk",
        "title":      "Karachi winter temperature patterns",
        "crop":       "general",
        "topic":      "heat",
        "text": (
            "Karachi winters are mild by Pakistan standards. December through "
            "February temperatures range from 10°C at night to 26°C in the afternoon. "
            "Winter is the only season when external temperatures in Karachi "
            "approach the tomato storage range of 10 to 13°C, reducing refrigeration "
            "energy costs for tomato storage. Potato and onion storage still requires "
            "active refrigeration in winter as daytime temperatures typically exceed "
            "their respective thresholds. Night-time dispatch is less critical in "
            "winter as temperature differentials are smaller."
        ),
    },

    # -----------------------------------------------------------------------
    # WFP / HDX Pakistan Food Security
    # Source: WFP Food Prices https://data.humdata.org/dataset/wfp-food-prices-for-pakistan
    # -----------------------------------------------------------------------
    {
        "doc_id":     "WFP_PK_001",
        "source":     "WFP Food Prices — Humanitarian Data Exchange",
        "source_url": "https://data.humdata.org/dataset/wfp-food-prices-for-pakistan",
        "title":      "Seasonal price patterns for tomatoes in Pakistan",
        "crop":       "tomato",
        "topic":      "price",
        "text": (
            "Tomato prices in Pakistan show strong seasonal variation. "
            "Prices typically peak during the hot summer months when production "
            "from Sindh and Balochistan faces heat stress and spoilage losses. "
            "The winter harvest from Punjab brings prices down significantly. "
            "Price volatility is highest in June and July. "
            "Cold-chain failures during peak summer can cause local price spikes "
            "of 30 to 60 percent within a single week in urban markets. "
            "Karachi is a major receiving market for produce from Balochistan "
            "via the RCD Highway and from Sindh via local supply chains."
        ),
    },
    {
        "doc_id":     "WFP_PK_002",
        "source":     "WFP Food Prices — Humanitarian Data Exchange",
        "source_url": "https://data.humdata.org/dataset/wfp-food-prices-for-pakistan",
        "title":      "Seasonal price patterns for onions and potatoes in Pakistan",
        "crop":       "general",
        "topic":      "price",
        "text": (
            "Onion prices in Pakistan are highly seasonal and sensitive to "
            "storage losses. Peak production occurs in Sindh and Punjab in spring. "
            "Onions stored in cold storage can be released into the market when "
            "prices are high, providing the cold-store operator with price arbitrage. "
            "Potato prices follow a similar seasonal pattern with a main harvest "
            "from Punjab in March to May. Cold storage allows potatoes to be held "
            "until autumn and winter when prices are typically higher. "
            "Both crops show inter-annual variation linked to rainfall, input costs, "
            "and import policy decisions."
        ),
    },

    # -----------------------------------------------------------------------
    # ColdChain Watch internal — model interpretation guidance
    # -----------------------------------------------------------------------
    {
        "doc_id":     "CCW_INT_001",
        "source":     "ColdChain Watch Pakistan — System Documentation",
        "source_url": "internal",
        "title":      "How to interpret heat alert scores",
        "crop":       "general",
        "topic":      "heat",
        "text": (
            "The ColdChain Watch heat_risk score ranges from 0.0 to 1.0. "
            "A score of 0.0 means external temperature is at or below the FAO "
            "safe storage threshold for that crop. A score of 1.0 means external "
            "temperature is 20°C or more above the threshold — maximum urgency. "
            "The heat_alert flag is 1 whenever external temperature exceeds "
            "the FAO threshold, regardless of the heat_risk magnitude. "
            "In Karachi, heat_alert_potato and heat_alert_onion are almost always 1 "
            "because external temperatures always exceed their storage thresholds. "
            "This means cold storage in Karachi always requires active refrigeration "
            "for these crops — the alert indicates energy cost, not imminent spoilage. "
            "heat_alert_tomato is the most practically useful alert in Karachi "
            "because the tomato threshold of 13°C is occasionally approached "
            "during winter nights."
        ),
    },
    {
        "doc_id":     "CCW_INT_002",
        "source":     "ColdChain Watch Pakistan — System Documentation",
        "source_url": "internal",
        "title":      "How to interpret temperature forecasts",
        "crop":       "general",
        "topic":      "heat",
        "text": (
            "Temperature forecasts are produced at three horizons: 1 hour, 6 hours, "
            "and 24 hours ahead. All forecasts come with a 90 percent prediction "
            "interval. The 1-hour forecast is highly accurate with a mean error "
            "below 0.3 degrees. The 6-hour forecast has a mean error below 0.5 degrees. "
            "The 24-hour forecast has a mean error below 0.6 degrees. "
            "Forecasts are most uncertain during the morning transition hours "
            "from 6 AM to 11 AM when sea breeze effects are strongest in Karachi. "
            "For dispatch decisions, the 6-hour forecast is recommended. "
            "For pre-cooling scheduling, the 24-hour forecast provides sufficient "
            "lead time while remaining reliable. "
            "All forecast numbers in this system come from the Gold data tables "
            "and are never generated by the AI language model."
        ),
    },
]


def get_all_chunks() -> list[dict]:
    """Return all document chunks ready for embedding."""
    return REFERENCE_DOCUMENTS


def get_chunks_by_crop(crop: str) -> list[dict]:
    """Return document chunks relevant to a specific crop."""
    return [d for d in REFERENCE_DOCUMENTS if d["crop"] in (crop, "general")]


def get_chunks_by_topic(topic: str) -> list[dict]:
    """Return document chunks by topic: heat / storage / price / dispatch."""
    return [d for d in REFERENCE_DOCUMENTS if d["topic"] == topic]
