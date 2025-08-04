"""
Geospatial Analysis of Voter and Property Data
Final Project Assignment

This script performs comprehensive geospatial analysis of voter registration 
and property data for North Carolina counties.

Requirements:
- geopandas
- pandas
- matplotlib
- seaborn
- scipy
- numpy
- requests (for data downloading)
- fuzzywuzzy (for address matching)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from thefuzz import fuzz, process
import requests
import os
import warnings
warnings.filterwarnings('ignore')

class VoterPropertyAnalysis:
    def __init__(self, county1="Pitt", county2="Johnston"):
        """
        Initialize the analysis with two NC counties
        Default: Pitt County (as required) and Johnston County
        """
        self.county1 = county1
        self.county2 = county2
        self.crs = "EPSG:2264"  # NAD83 / North Carolina (ftUS)
        self.data = {}
        
    def download_data(self):
        """
        Download required datasets from NC OneMap and NC State Board of Elections
        Note: This function provides the framework - actual downloads may require 
        manual steps due to website authentication requirements
        """
        print("=== DATA COLLECTION PHASE ===")
        print("Please manually download the following datasets:")
        print(f"1. From NC OneMap (https://www.nconemap.gov/):")
        print(f"   - Addresses dataset for {self.county1} County")
        print(f"   - Parcels dataset for {self.county1} County")
        print(f"   - Addresses dataset for {self.county2} County")
        print(f"   - Parcels dataset for {self.county2} County")
        print(f"2. From NC State Board of Elections:")
        print(f"   - Voter registration data for {self.county1} County")
        print(f"   - Voter registration data for {self.county2} County")
        print("\nPlace downloaded files in the current directory with naming convention:")
        print("- {county}_addresses.shp")
        print("- {county}_parcels.shp") 
        print("- {county}_voters.txt (tab-delimited)")
        
    def load_data(self):
        """Load all datasets for both counties"""
        print("\n=== LOADING DATA ===")
        
        counties = [self.county1, self.county2]
        for county in counties:
            print(f"Loading data for {county} County...")
            
            try:
                # Load spatial data
                addresses_file = f"{county.lower()}_addresses.shp"
                parcels_file = f"{county.lower()}_parcels.shp"
                voters_file = f"{county.lower()}_voters.txt"
                
                if os.path.exists(addresses_file):
                    addresses = gpd.read_file(addresses_file)
                    addresses = addresses.to_crs(self.crs)
                    print(f"  - Loaded {len(addresses)} addresses")
                else:
                    print(f"  - Warning: {addresses_file} not found")
                    addresses = None
                
                if os.path.exists(parcels_file):
                    parcels = gpd.read_file(parcels_file)
                    parcels = parcels.to_crs(self.crs)
                    print(f"  - Loaded {len(parcels)} parcels")
                else:
                    print(f"  - Warning: {parcels_file} not found")
                    parcels = None
                
                if os.path.exists(voters_file):
                    voters = pd.read_csv(voters_file, sep='\t', low_memory=False)
                    print(f"  - Loaded {len(voters)} voter records")
                else:
                    print(f"  - Warning: {voters_file} not found")
                    voters = None
                    
                self.data[county] = {
                    'addresses': addresses,
                    'parcels': parcels,
                    'voters': voters
                }
                
            except Exception as e:
                print(f"Error loading data for {county}: {e}")
                
    def geocode_voters(self, county):
        """
        Geocode voter addresses using the addresses dataset
        """
        print(f"\n=== GEOCODING VOTERS FOR {county.upper()} COUNTY ===")
        
        voters = self.data[county]['voters']
        addresses = self.data[county]['addresses']
        
        if voters is None or addresses is None:
            print(f"Missing data for {county} county")
            return None
            
        # Clean and prepare address fields for matching
        # Common voter registration fields: res_street_address, res_city_desc
        # Common address dataset fields: FULLADDR, STREET, CITY
        
        voter_cols = voters.columns.tolist()
        addr_cols = addresses.columns.tolist()
        
        print("Available voter columns:", [col for col in voter_cols if 'addr' in col.lower() or 'street' in col.lower()][:5])
        print("Available address columns:", [col for col in addr_cols if 'addr' in col.lower() or 'street' in col.lower()][:5])
        
        # Create standardized address strings for matching
        voters['full_address'] = self._create_voter_address_string(voters)
        addresses['full_address'] = self._create_address_string(addresses)
        
        # Perform fuzzy matching
        geocoded_voters = []
        successful_matches = 0
        
        print(f"Attempting to geocode {len(voters)} voters...")
        
        for idx, voter in voters.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx} voters ({successful_matches} matched)")
                
            voter_addr = voter['full_address']
            if pd.isna(voter_addr) or len(voter_addr) < 5:
                continue
                
            # Find best match using fuzzy string matching
            match = process.extractOne(
                voter_addr, 
                addresses['full_address'].tolist(),
                scorer=fuzz.ratio
            )
            
            if match and match[1] > 80:  # 80% similarity threshold
                match_idx = addresses[addresses['full_address'] == match[0]].index[0]
                matched_addr = addresses.loc[match_idx]
                
                # Add coordinates to voter record
                voter_with_coords = voter.copy()
                voter_with_coords['geometry'] = matched_addr['geometry']
                voter_with_coords['match_score'] = match[1]
                geocoded_voters.append(voter_with_coords)
                successful_matches += 1
        
        # Convert to GeoDataFrame
        if geocoded_voters:
            geocoded_gdf = gpd.GeoDataFrame(geocoded_voters, crs=self.crs)
            success_rate = (successful_matches / len(voters)) * 100
            
            print(f"Successfully geocoded {successful_matches} out of {len(voters)} voters")
            print(f"Success rate: {success_rate:.1f}%")
            
            # Save to GeoPackage
            output_file = f"{county.lower()}_geocoded_voters.gpkg"
            geocoded_gdf.to_file(output_file, driver='GPKG')
            print(f"Saved geocoded voters to {output_file}")
            
            self.data[county]['geocoded_voters'] = geocoded_gdf
            self.data[county]['geocoding_success_rate'] = success_rate
            
            return geocoded_gdf
        else:
            print("No successful matches found")
            return None
    
    def _create_voter_address_string(self, voters):
        """Create standardized address string from voter data"""
        # Common voter fields - adjust based on actual column names
        addr_fields = [
            'res_street_address',
            'mail_addr1',
            'mail_addr2',
            'mail_addr3',
            'mail_addr4',
        ]
        city_fields = ['res_city_desc', 'mail_city', 'res_city']
        
        addr_col = None
        city_col = None
        
        for field in addr_fields:
            if field in voters.columns:
                addr_col = field
                break
                
        for field in city_fields:
            if field in voters.columns:
                city_col = field
                break
        
        if addr_col:
            if city_col:
                return voters[addr_col].astype(str) + " " + voters[city_col].astype(str)
            else:
                return voters[addr_col].astype(str)
        else:
            return pd.Series([''] * len(voters))   

    def _create_address_string(self, addresses):
        # Try Pitt-style fields
        pitt_parts = ['addr_hn', 'addr_sn', 'addr_st']
        if all(part in addresses.columns for part in pitt_parts):
            return pd.Series([
                f"{hn} {sn} {st}".strip()
                for hn, sn, st in zip(addresses['addr_hn'], addresses['addr_sn'], addresses['addr_st'])
            ], index=addresses.index)

        # Try Johnston-style fields
        johnston_parts = ['Add_Number', 'St_Name']
        if all(part in addresses.columns for part in johnston_parts):
            return pd.Series([
                f"{num} {name}".strip()
                for num, name in zip(addresses['Add_Number'], addresses['St_Name'])
            ], index=addresses.index)

        # Try single fallback field
        fallback_fields = ['Full_Addre', 'full_addre', 'St_Address']
        for field in fallback_fields:
            if field in addresses.columns:
                return addresses[field].astype(str).fillna("")

        # If no usable fields found
        print("Warning: No usable address fields found for matching.")
        return pd.Series([""] * len(addresses), index=addresses.index)

    

    def spatial_join_parcels(self, county):
        """Join geocoded voters with parcel data"""
        print(f"\n=== SPATIAL JOIN WITH PARCELS FOR {county.upper()} ===")
        
        geocoded_voters = self.data[county].get('geocoded_voters')
        parcels = self.data[county]['parcels']
        
        if geocoded_voters is None or parcels is None:
            print(f"Missing required data for {county}")
            return None
        
        # Perform spatial join
        voters_with_parcels = gpd.sjoin(geocoded_voters, parcels, how='left', predicate='within')
        
        successful_joins = len(voters_with_parcels.dropna(subset=['index_right']))
        join_rate = (successful_joins / len(voters_with_parcels)) * 100
        
        print(f"Successfully joined {successful_joins} out of {len(voters_with_parcels)} voters with parcels")
        print(f"Join success rate: {join_rate:.1f}%")
        
        self.data[county]['voters_with_parcels'] = voters_with_parcels
        return voters_with_parcels
    
    def analyze_political_property_values(self):
        """
        Analysis #1: Investigate relationship between political affiliation and property values
        """
        print("\n=== ANALYSIS 1: POLITICAL AFFILIATION vs PROPERTY VALUES ===")
        
        results = {}
        
        for county in [self.county1, self.county2]:
            print(f"\nAnalyzing {county} County...")
            
            voters_parcels = self.data[county].get('voters_with_parcels')
            if voters_parcels is None:
                continue
            
            # Find party affiliation and property value columns
            party_cols = [col for col in voters_parcels.columns if 'party' in col.lower()]
            value_cols = [col for col in voters_parcels.columns if 'parval' in col.upper() or 'value' in col.lower()]
            
            if not party_cols or not value_cols:
                print(f"Missing party or value columns for {county}")
                continue
                
            party_col = party_cols[0]
            value_col = value_cols[0]
            
            # Clean data
            df = voters_parcels[[party_col, value_col]].copy()
            df = df.dropna()
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna()
            df = df[df[value_col] > 0]  # Remove zero/negative values
            
            # Filter to main parties
            main_parties = ['DEM', 'REP', 'UNA']
            df = df[df[party_col].isin(main_parties)]
            
            if len(df) == 0:
                print(f"No valid data for {county}")
                continue
            
            # Calculate statistics
            party_stats = df.groupby(party_col)[value_col].agg(['count', 'mean', 'median', 'std'])
            
            print(f"\nProperty Value Statistics by Party ({county}):")
            print(party_stats)
            
            # Perform statistical tests
            dem_values = df[df[party_col] == 'DEM'][value_col]
            rep_values = df[df[party_col] == 'REP'][value_col]
            una_values = df[df[party_col] == 'UNA'][value_col]
            
            # ANOVA test
            if len(dem_values) > 0 and len(rep_values) > 0 and len(una_values) > 0:
                f_stat, p_value = stats.f_oneway(dem_values, rep_values, una_values)
                print(f"\nANOVA Test Results:")
                print(f"F-statistic: {f_stat:.4f}")
                print(f"P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print("Significant differences found between parties (p < 0.05)")
                else:
                    print("No significant differences found between parties (p >= 0.05)")
            
            results[county] = {
                'stats': party_stats,
                'data': df,
                'party_col': party_col,
                'value_col': value_col
            }
        
        # Create visualizations
        self._plot_property_values_by_party(results)
        return results
    
    def _plot_property_values_by_party(self, results):
        """Create visualizations for property value analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Property Values by Political Affiliation', fontsize=16)
        
        for i, county in enumerate([self.county1, self.county2]):
            if county not in results:
                continue
                
            data = results[county]['data']
            value_col = results[county]['value_col']
            party_col = results[county]['party_col']
            
            # Box plot
            ax1 = axes[i, 0]
            data.boxplot(column=value_col, by=party_col, ax=ax1)
            ax1.set_title(f'{county} County - Property Values by Party')
            ax1.set_ylabel('Property Value ($)')
            
            # Bar plot of means
            ax2 = axes[i, 1]
            party_means = data.groupby(party_col)[value_col].mean()
            party_means.plot(kind='bar', ax=ax2)
            ax2.set_title(f'{county} County - Mean Property Values')
            ax2.set_ylabel('Mean Property Value ($)')
            ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('property_values_by_party.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_urban_rural_differences(self):
        """
        Analysis #2: Compare voter statistics between urban and rural areas
        """
        print("\n=== ANALYSIS 2: URBAN vs RURAL VOTER CHARACTERISTICS ===")
        
        for county in [self.county1, self.county2]:
            voters_parcels = self.data[county].get('voters_with_parcels')
            if voters_parcels is None:
                continue
                
            print(f"\nAnalyzing {county} County...")
            
            # Define urban areas using population density or parcel density
            # Calculate density using spatial analysis
            centroids = voters_parcels.geometry.centroid
            
            # Create buffers around each point and count neighbors
            buffer_distance = 1000  # 1000 feet
            voters_parcels['neighbor_count'] = 0
            
            for idx, point in centroids.items():
                buffer = point.buffer(buffer_distance)
                neighbors = centroids.within(buffer).sum() - 1  # Exclude self
                voters_parcels.loc[idx, 'neighbor_count'] = neighbors
            
            # Classify as urban/rural based on neighbor density
            urban_threshold = voters_parcels['neighbor_count'].quantile(0.7)
            voters_parcels['area_type'] = voters_parcels['neighbor_count'].apply(
                lambda x: 'Urban' if x >= urban_threshold else 'Rural'
            )
            
            # Find party column
            party_cols = [col for col in voters_parcels.columns if 'party' in col.lower()]
            if not party_cols:
                continue
            party_col = party_cols[0]
            
            # Analyze differences
            area_party_crosstab = pd.crosstab(
                voters_parcels['area_type'], 
                voters_parcels[party_col], 
                normalize='index'
            ) * 100
            
            print(f"Party Distribution by Area Type (%):")
            print(area_party_crosstab)
            
            # Save results
            output_file = f"{county.lower()}_urban_rural_analysis.gpkg"
            voters_parcels[['geometry', 'area_type', party_col, 'neighbor_count']].to_file(
                output_file, driver='GPKG'
            )
    
    def analyze_proximity_to_schools(self):
        """
        Analysis #3: Analyze voter characteristics by proximity to schools
        Uses statewide school dataset and filters by COUNTY
        """
        print("\n=== ANALYSIS 3: PROXIMITY TO SCHOOLS ===")

        school_file = "all_nc_schools.shp"

        if not os.path.exists(school_file):
            print(f"Statewide school file not found: {school_file}")
            return

        # Load and reproject the full schools file once
        all_schools = gpd.read_file(school_file).to_crs(self.crs)

        for county in [self.county1, self.county2]:
            print(f"\nAnalyzing school proximity in {county} County...")

            voters_parcels = self.data[county].get('voters_with_parcels')
            if voters_parcels is None:
                print(f"  - Voter data missing for {county}, skipping.")
                continue

            # Filter schools by county
            matching_schools = all_schools[all_schools['COUNTY'].str.upper() == county.upper()]
            if matching_schools.empty:
                print(f"  - No schools found for {county}")
                continue

            # Calculate distance to nearest school
            voters_parcels['dist_to_school'] = voters_parcels.geometry.apply(
                lambda voter_geom: matching_schools.distance(voter_geom).min()
            )

            # Optional: classify voters by proximity bands
            voters_parcels['school_proximity'] = pd.cut(
                voters_parcels['dist_to_school'],
                bins=[0, 1000, 5280, 15840, np.inf],  # under 1k ft, under 1 mi, under 3 mi, far
                labels=["<1000 ft", "≤1 mile", "≤3 miles", ">3 miles"]
            )

            # Analyze voting patterns by proximity
            party_cols = [col for col in voters_parcels.columns if 'party' in col.lower()]
            if not party_cols:
                print(f"  - Party column not found for {county}")
                continue

            party_col = party_cols[0]
            proximity_stats = pd.crosstab(
                voters_parcels['school_proximity'], voters_parcels[party_col],
                normalize='index') * 100

            print(f"  - Voter party distribution by proximity to schools in {county}:")
            print(proximity_stats.round(1))

            # Save result layer
            output_file = f"{county.lower()}_school_proximity_analysis.gpkg"
            voters_parcels[['geometry', party_col, 'dist_to_school', 'school_proximity']].to_file(
                output_file, driver='GPKG')
            print(f"  - Saved results to {output_file}")

            """
            Analysis #3: Analyze voter characteristics by proximity to schools
            Note: Requires school location data
            """
            print("\n=== ANALYSIS 3: PROXIMITY TO SCHOOLS ===")
            print("Note: This analysis requires school location data.")
            print("Please download school locations from county GIS websites or OpenStreetMap.")
            
            # Framework for school proximity analysis
            for county in [self.county1, self.county2]:
                school_file = f"{county.lower()}_schools.shp"
                
                if os.path.exists(school_file):
                    schools = gpd.read_file(school_file).to_crs(self.crs)
                    voters_parcels = self.data[county].get('voters_with_parcels')
                    
                    if voters_parcels is not None:
                        # Calculate distance to nearest school
                        voters_parcels['dist_to_school'] = voters_parcels.geometry.apply(
                            lambda x: schools.geometry.distance(x).min()
                        )
                        
                        # Analyze voting patterns by school proximity
                        # (Implementation would continue here)
                        print(f"School proximity analysis completed for {county}")
                else:
                    print(f"School data not found for {county}: {school_file}")
    
    def analyze_age_distribution(self):
        """
        Analysis #4: Age distribution analysis
        """
        print("\n=== ANALYSIS 4: VOTER AGE DISTRIBUTION ===")
        
        for county in [self.county1, self.county2]:
            voters = self.data[county]['voters']
            if voters is None:
                continue
                
            # Find birth date or age columns
            birth_cols = [col for col in voters.columns if 'birth' in col.lower()]
            age_cols = [col for col in voters.columns if 'age' in col.lower()]
            
            if birth_cols:
                birth_col = birth_cols[0]
                # Calculate age from birth date
                voters['birth_date'] = pd.to_datetime(voters[birth_col], errors='coerce')
                current_year = pd.Timestamp.now().year
                voters['age'] = current_year - voters['birth_date'].dt.year
            elif age_cols:
                voters['age'] = pd.to_numeric(voters[age_cols[0]], errors='coerce')
            else:
                print(f"No age data found for {county}")
                continue
            
            # Age group analysis
            voters['age_group'] = pd.cut(
                voters['age'], 
                bins=[0, 30, 50, 65, 100], 
                labels=['18-30', '31-50', '51-65', '65+']
            )
            
            party_cols = [col for col in voters.columns if 'party' in col.lower()]
            if party_cols:
                party_col = party_cols[0]
                age_party_crosstab = pd.crosstab(
                    voters['age_group'], 
                    voters[party_col], 
                    normalize='index'
                ) * 100
                
                print(f"\n{county} County - Party Distribution by Age Group (%):")
                print(age_party_crosstab)
    
    def analyze_voter_turnout_by_precinct(self):
        """
        Analysis #5: Voter turnout analysis by precinct/district
        """
        print("\n=== ANALYSIS 5: VOTER TURNOUT BY PRECINCT ===")
        
        for county in [self.county1, self.county2]:
            voters = self.data[county]['voters']
            if voters is None:
                continue
                
            # Find precinct and voting history columns
            precinct_cols = [col for col in voters.columns if 'precinct' in col.lower() or 'district' in col.lower()]
            
            if not precinct_cols:
                print(f"No precinct data found for {county}")
                continue
                
            precinct_col = precinct_cols[0]
            
            # Calculate registration rates by precinct
            precinct_stats = voters.groupby(precinct_col).agg({
                'party_cd': 'count',  # Assuming party_cd exists    
            }).rename(columns={'party_cd': 'total_registered'})
            
            print(f"\n{county} County - Top 10 Precincts by Registration:")
            print(precinct_stats.sort_values('total_registered', ascending=False).head(10))
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\n" + "="*60)
        print("GEOSPATIAL VOTER AND PROPERTY ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        for county in [self.county1, self.county2]:
            print(f"\n{county.upper()} COUNTY SUMMARY:")
            print("-" * 30)
            
            if county in self.data:
                data = self.data[county]
                
                # Data loading summary
                if data.get('voters') is not None:
                    print(f"Voter Records: {len(data['voters']):,}")
                if data.get('addresses') is not None:
                    print(f"Address Records: {len(data['addresses']):,}")
                if data.get('parcels') is not None:
                    print(f"Parcel Records: {len(data['parcels']):,}")
                
                # Geocoding success rate
                if 'geocoding_success_rate' in data:
                    print(f"Geocoding Success Rate: {data['geocoding_success_rate']:.1f}%")
                
                # Voter demographics summary
                if data.get('geocoded_voters') is not None:
                    gv = data['geocoded_voters']
                    party_cols = [col for col in gv.columns if 'party' in col.lower()]
                    if party_cols:
                        party_dist = gv[party_cols[0]].value_counts()
                        print(f"Party Distribution:")
                        for party, count in party_dist.head(5).items():
                            print(f"  {party}: {count:,} ({count/len(gv)*100:.1f}%)")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Geospatial Voter and Property Analysis...")
        print(f"Counties: {self.county1} and {self.county2}")
        print(f"Spatial Reference: {self.crs}")
        
        # Data collection guidance
        self.download_data()
        
        # Load data
        self.load_data()
        
        # Geocoding
        for county in [self.county1, self.county2]:
            self.geocode_voters(county)
            self.spatial_join_parcels(county)
        
        # Spatial analyses
        self.analyze_political_property_values()
        self.analyze_urban_rural_differences()
        self.analyze_proximity_to_schools()
        self.analyze_age_distribution()
        self.analyze_voter_turnout_by_precinct()
        
        # Final report
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("Generated files:")
        print("- {county}_geocoded_voters.gpkg (for each county)")
        print("- {county}_urban_rural_analysis.gpkg (for each county)")
        print("- property_values_by_party.png")
        print("="*60)


# Example usage and execution
if __name__ == "__main__":
    # Initialize analysis for Pitt County (required) and Johnston County
    analysis = VoterPropertyAnalysis(county1="Pitt", county2="Johnston")
    
    # For demonstration purposes, show key methods
    print("Geospatial Voter and Property Analysis System")
    print("=" * 50)
    print("\nThis script provides a complete framework for:")
    print("1. Data collection from NC OneMap and NC Board of Elections")
    print("2. Geocoding voter addresses using fuzzy string matching")
    print("3. Spatial joins with parcel data")
    print("4. Multiple spatial analyses:")
    print("   - Political affiliation vs property values")
    print("   - Urban vs rural voter characteristics")
    print("   - Proximity to schools analysis")
    print("   - Age distribution analysis")
    print("   - Voter turnout by precinct")
    print("5. Comprehensive reporting and visualization")
    
    print("\nTo run the complete analysis:")
    print("1. Download required datasets (guidance provided)")
    print("2. Execute: analysis.run_complete_analysis()")
    
    # Uncomment the following line to run the complete analysis
    analysis.run_complete_analysis()
