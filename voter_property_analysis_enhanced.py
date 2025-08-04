# Cleaned and optimized script based on your original
# Includes: deduplicated methods, proper structure, and faster fuzzy matcher using indexing

# Due to length, this will be modularized across multiple parts if needed

# Note: Will use `rapidfuzz` for performance if installed. Fallback remains `fuzz`.

# Start of cleaned script

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from thefuzz import fuzz, process  # fallback
import os
import warnings
from shapely.strtree import STRtree

try:
    from rapidfuzz import process as rprocess
    FAST_MATCH = True
except ImportError:
    FAST_MATCH = False

warnings.filterwarnings("ignore")


class VoterPropertyAnalysis:
    def __init__(self, county1="Pitt", county2="Johnston"):
        self.county1 = county1
        self.county2 = county2
        self.crs = "EPSG:2264"
        self.data = {}

    def _create_address_string(self, addresses):
        pitt_parts = ['addr_hn', 'addr_sn', 'addr_st']
        if all(part in addresses.columns for part in pitt_parts):
            return addresses[pitt_parts].fillna('').agg(' '.join, axis=1).str.strip()

        johnston_parts = ['Add_Number', 'St_Name']
        if all(part in addresses.columns for part in johnston_parts):
            return addresses[johnston_parts].fillna('').agg(' '.join, axis=1).str.strip()

        fallback_fields = ['Full_Addre', 'full_addre', 'St_Address']
        for field in fallback_fields:
            if field in addresses.columns:
                return addresses[field].astype(str).fillna("")

        print("Warning: No usable address fields found for matching.")
        return pd.Series([""] * len(addresses), index=addresses.index)

    def _create_voter_address_string(self, voters):
        addr_fields = ['res_street_address', 'mail_addr1', 'mail_addr2', 'mail_addr3']
        city_fields = ['res_city_desc', 'mail_city', 'res_city']
        addr_col = next((col for col in addr_fields if col in voters.columns), None)
        city_col = next((col for col in city_fields if col in voters.columns), None)

        if addr_col:
            if city_col:
                return voters[addr_col].astype(str) + " " + voters[city_col].astype(str)
            else:
                return voters[addr_col].astype(str)
        return pd.Series([""] * len(voters))

    def geocode_voters(self, county):
        print(f"\n=== GEOCODING VOTERS FOR {county.upper()} COUNTY ===")
        voters = self.data[county]['voters']
        addresses = self.data[county]['addresses']

        if voters is None or addresses is None:
            print(f"Missing data for {county} county")
            return None

        voters['full_address'] = self._create_voter_address_string(voters)
        addresses['full_address'] = self._create_address_string(addresses)

        address_list = addresses['full_address'].tolist()
        address_lookup = dict(zip(address_list, addresses.index))

        matcher = rprocess if FAST_MATCH else process
        geocoded = []
        matches = 0

        for idx, voter in voters.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx} voters ({matches} matched)")

            vaddr = voter['full_address']
            if not vaddr or pd.isna(vaddr):
                continue

            match = matcher.extractOne(vaddr, address_list, scorer=fuzz.ratio)

            if match and match[1] > 80:
                matched_idx = address_lookup[match[0]]
                matched_row = addresses.loc[matched_idx]
                voter_with_coords = voter.copy()
                voter_with_coords['geometry'] = matched_row['geometry']
                voter_with_coords['match_score'] = match[1]
                geocoded.append(voter_with_coords)
                matches += 1

        if geocoded:
            geocoded_gdf = gpd.GeoDataFrame(geocoded, crs=self.crs)
            rate = matches / len(voters) * 100
            print(f"Matched {matches} of {len(voters)} ({rate:.1f}%)")
            out = f"{county.lower()}_geocoded_voters.gpkg"
            geocoded_gdf.to_file(out, driver='GPKG')
            print(f"Saved to {out}")
            self.data[county]['geocoded_voters'] = geocoded_gdf
            self.data[county]['geocoding_success_rate'] = rate
            return geocoded_gdf

        print("No matches found.")
        return None

# Continued: Remaining methods for VoterPropertyAnalysis class

    def load_data(self):
        print("\n=== LOADING DATA ===")
        for county in [self.county1, self.county2]:
            print(f"Loading data for {county} County...")
            try:
                ad_file = f"{county.lower()}_addresses.shp"
                pa_file = f"{county.lower()}_parcels.shp"
                vo_file = f"{county.lower()}_voters.txt"

                addresses = gpd.read_file(ad_file).to_crs(self.crs) if os.path.exists(ad_file) else None
                parcels = gpd.read_file(pa_file).to_crs(self.crs) if os.path.exists(pa_file) else None
                voters = pd.read_csv(vo_file, sep='\t', low_memory=False) if os.path.exists(vo_file) else None

                self.data[county] = {
                    'addresses': addresses,
                    'parcels': parcels,
                    'voters': voters
                }

                print(f"  - {len(addresses) if addresses is not None else 0} addresses")
                print(f"  - {len(parcels) if parcels is not None else 0} parcels")
                print(f"  - {len(voters) if voters is not None else 0} voter records")

            except Exception as e:
                print(f"Error loading {county} data: {e}")

    def spatial_join_parcels(self, county):
        print(f"\n=== SPATIAL JOIN WITH PARCELS FOR {county.upper()} ===")
        voters = self.data[county].get('geocoded_voters')
        parcels = self.data[county].get('parcels')

        if voters is None or parcels is None:
            print(f"Missing data for {county}")
            return None

        joined = gpd.sjoin(voters, parcels, how='left', predicate='within')
        joined_count = joined.dropna(subset=['index_right']).shape[0]
        rate = joined_count / len(joined) * 100
        print(f"  - {joined_count} of {len(joined)} joined ({rate:.1f}%)")

        self.data[county]['voters_with_parcels'] = joined
        return joined

    def analyze_proximity_to_schools(self):
        print("\n=== ANALYSIS 3: PROXIMITY TO SCHOOLS ===")
        school_file = "all_nc_schools.shp"
        if not os.path.exists(school_file):
            print(f"Missing school file: {school_file}")
            return

        schools_all = gpd.read_file(school_file).to_crs(self.crs)

        for county in [self.county1, self.county2]:
            voters = self.data[county].get('voters_with_parcels')
            if voters is None:
                continue

            local_schools = schools_all[schools_all['COUNTY'].str.upper() == county.upper()]
            if local_schools.empty:
                print(f"  - No schools found for {county}")
                continue

            voters['dist_to_school'] = voters.geometry.apply(lambda x: local_schools.distance(x).min())
            voters['school_proximity'] = pd.cut(voters['dist_to_school'],
                bins=[0, 1000, 5280, 15840, np.inf],
                labels=["<1000 ft", "≤1 mile", "≤3 miles", ">3 miles"]
            )

            party_col = next((col for col in voters.columns if 'party' in col.lower()), None)
            if not party_col:
                continue

            proximity_summary = pd.crosstab(voters['school_proximity'], voters[party_col], normalize='index') * 100
            print(f"\n{county} County - Party % by School Proximity:")
            print(proximity_summary.round(1))

            out = f"{county.lower()}_school_proximity_analysis.gpkg"
            voters[['geometry', party_col, 'dist_to_school', 'school_proximity']].to_file(out, driver='GPKG')

    def run_complete_analysis(self):
        print("Starting Geospatial Voter and Property Analysis...")
        self.load_data()
        for county in [self.county1, self.county2]:
            self.geocode_voters(county)
            self.spatial_join_parcels(county)
        self.analyze_proximity_to_schools()


# Continued: Add full analysis methods

    def analyze_political_property_values(self):
        print("\n=== ANALYSIS 1: PARTY vs PROPERTY VALUE ===")
        results = {}
        for county in [self.county1, self.county2]:
            df = self.data[county].get('voters_with_parcels')
            if df is None:
                continue

            party_col = next((c for c in df.columns if 'party' in c.lower()), None)
            value_col = next((c for c in df.columns if 'parval' in c.lower() or 'value' in c.lower()), None)
            if not party_col or not value_col:
                print(f"Missing required fields for {county}")
                continue

            df = df[[party_col, value_col]].copy().dropna()
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna()
            df = df[df[value_col] > 0]
            df = df[df[party_col].isin(['DEM', 'REP', 'UNA'])]

            if df.empty:
                continue

            print(f"\n{county} County Summary:")
            stats = df.groupby(party_col)[value_col].agg(['count', 'mean', 'median'])
            print(stats.round(0))

            from scipy import stats as s
            fval, pval = s.f_oneway(*[df[df[party_col] == p][value_col] for p in ['DEM', 'REP', 'UNA']])
            print(f"ANOVA P-value: {pval:.4f}")
            results[county] = df

        return results

    def analyze_urban_rural_differences(self):
        print("\n=== ANALYSIS 2: URBAN vs RURAL ===")
        for county in [self.county1, self.county2]:
            df = self.data[county].get('voters_with_parcels')
            if df is None:
                continue
            df = df.copy()
            df['neighbor_count'] = df.geometry.apply(
                lambda pt: df.geometry.distance(pt).lt(1000).sum() - 1)
            threshold = df['neighbor_count'].quantile(0.7)
            df['area_type'] = df['neighbor_count'].apply(lambda n: 'Urban' if n >= threshold else 'Rural')

            party_col = next((c for c in df.columns if 'party' in c.lower()), None)
            if party_col:
                crosstab = pd.crosstab(df['area_type'], df[party_col], normalize='index') * 100
                print(f"\n{county} - Area Type by Party %:\n", crosstab.round(1))

            out = f"{county.lower()}_urban_rural_analysis.gpkg"
            df[['geometry', 'area_type', party_col]].to_file(out, driver='GPKG')

    def analyze_age_distribution(self):
        print("\n=== ANALYSIS 4: AGE DISTRIBUTION ===")
        for county in [self.county1, self.county2]:
            df = self.data[county].get('voters')
            if df is None:
                continue
            df = df.copy()
            birth_col = next((c for c in df.columns if 'birth' in c.lower()), None)
            if birth_col:
                df['birth_date'] = pd.to_datetime(df[birth_col], errors='coerce')
                df['age'] = pd.Timestamp.now().year - df['birth_date'].dt.year
            elif 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
            else:
                continue

            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 120], labels=['18-30', '31-50', '51-65', '65+'])
            party_col = next((c for c in df.columns if 'party' in c.lower()), None)
            if party_col:
                summary = pd.crosstab(df['age_group'], df[party_col], normalize='index') * 100
                print(f"\n{county} - Party % by Age Group:\n", summary.round(1))

    def analyze_voter_turnout_by_precinct(self):
        print("\n=== ANALYSIS 5: TURNOUT BY PRECINCT ===")
        for county in [self.county1, self.county2]:
            df = self.data[county].get('voters')
            if df is None:
                continue
            df = df.copy()
            precinct_col = next((c for c in df.columns if 'precinct' in c.lower() or 'district' in c.lower()), None)
            if not precinct_col or 'party_cd' not in df.columns:
                continue
            summary = df.groupby(precinct_col).size().rename('total_voters')
            print(f"\n{county} - Top Precincts:\n", summary.sort_values(ascending=False).head(10))

    def create_summary_report(self):
        print("\n=== FINAL SUMMARY ===")
        for county in [self.county1, self.county2]:
            print(f"\n{county.upper()} COUNTY")
            data = self.data[county]
            print(f"  Voters: {len(data.get('voters', [])):,}")
            print(f"  Geocoded: {len(data.get('geocoded_voters', [])):,}")
            print(f"  Parcels: {len(data.get('parcels', [])):,}")
            print(f"  Address Records: {len(data.get('addresses', [])):,}")


# Example usage
if __name__ == "__main__":
    analysis = VoterPropertyAnalysis("Pitt", "Johnston")
    analysis.run_complete_analysis()
