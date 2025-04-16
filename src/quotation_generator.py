import hashlib
import json
import os

class QuotationGenerator:
    def __init__(self, pricing_rules=None, counts_file='quotation_counts.json'):
        # Pricing rules: dict mapping class_id to price per unit
        if pricing_rules is None:
            self.pricing_rules = {
                0: 10.0,  # switch
                1: 20.0,  # light
                2: 30.0,  # electrical outlet
            }
        else:
            self.pricing_rules = pricing_rules

        self.counts_file = counts_file
        # Load or initialize quotation counts
        if os.path.exists(self.counts_file):
            with open(self.counts_file, 'r') as f:
                self.quotation_counts = json.load(f)
        else:
            self.quotation_counts = {}

    def generate_quotation(self, detections):
        """
        Generate a bill of materials and total cost based on detections.

        Args:
            detections (list of dict): Each dict contains 'class_id', 'confidence', 'bbox'

        Returns:
            dict: {
                'items': {class_id: quantity},
                'total_cost': float,
                'unit_prices': dict,
                'quotation_hash': str,
                'download_count': int
            }
        """
        items = {}
        for det in detections:
            class_id = det['class_id']
            items[class_id] = items.get(class_id, 0) + 1

        total_cost = 0.0
        for class_id, quantity in items.items():
            price_per_unit = self.pricing_rules.get(class_id, 0.0)
            total_cost += price_per_unit * quantity

        # Generate a unique hash for the quotation based on items and total_cost
        hash_input = json.dumps({'items': items, 'total_cost': total_cost}, sort_keys=True).encode('utf-8')
        quotation_hash = hashlib.sha256(hash_input).hexdigest()

        # Update download count
        count = self.quotation_counts.get(quotation_hash, 0) + 1
        self.quotation_counts[quotation_hash] = count

        # Save updated counts
        with open(self.counts_file, 'w') as f:
            json.dump(self.quotation_counts, f, indent=2)

        return {
            'items': items,
            'total_cost': total_cost,
            'unit_prices': self.pricing_rules,
            'quotation_hash': quotation_hash,
            'download_count': count
        }
