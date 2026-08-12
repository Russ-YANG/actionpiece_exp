# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset classes for GenRec, imported lazily by name."""


def __getattr__(name: str):
  if name == 'AmazonReviews2014':
    from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
    return AmazonReviews2014
  if name == 'NineRec':
    from genrec.datasets.NineRec.dataset import NineRec
    return NineRec
  raise AttributeError(name)


__all__ = ['AmazonReviews2014', 'NineRec']
